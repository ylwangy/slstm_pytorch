import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder

from fairseq.modules import LayerNorm, PositionalEmbedding
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


from fairseq.modules.fairseq_dropout import FairseqDropout

from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import SlstmHubInterface
from fairseq.modules.quant_noise import quant_noise

logger = logging.getLogger(__name__)


@register_model("slstm")
class SlstmModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {}

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = SlstmEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, xleft, xright, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)
        # print(x.size())
        # print(extra.size())
        if classification_head_name is not None:
            extra = self.classification_heads[classification_head_name](extra)

        return x, xleft, xright, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = SlstmClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )


    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        # logger.info(x["args"])
        return SlstmHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class SlstmLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation

        # print('------')
        # print(features.size())   #bsz seqlen dim
        # print(masked_tokens)
        # print(masked_tokens.size()) #bsz seqlen
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        # print(features.size())  #mask size  dim
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class SlstmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout=0.1,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        # self.out_proj = apply_quant_noise_(
        #     nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        # )
        # if do_spectral_norm:
        #     if q_noise != 0:
        #         raise NotImplementedError(
        #             "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
        #         )
        #     self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        ############################## already use global node
        x = self.dropout(features)
        x = self.classifier(x)
        # x = self.activation_fn(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


class SlstmEncoder(FairseqEncoder):
    """SLSTM encoder."""
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        base_architecture(args)
        self.args = args
        embed_tokens = self.build_embedding(len(dictionary), args.encoder_embed_dim, dictionary.pad())
        self.embed_tokens = embed_tokens
        
        # self.emb_init()
        
        self.dropout = args.dropout
        # self.encoder_layerdrop = args.encoder_layerdrop
        self.slstm_kernel_size = args.slstm_kernel_size

        self.padding_idx = embed_tokens.padding_idx
        self.pos_g2h = args.pos_g2h
        
        self.pos_h2g = args.pos_h2g
        self.use_layer_norm = args.use_layer_norm
        self.mt = args.mt
        
        self.pos_input = args.pos_input

        embed_dim = embed_tokens.embedding_dim


        self.max_source_positions = args.max_source_positions


        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings and self.pos_input
            else None
        )

        self.encoder = self.build_encoder(args,dictionary)
        self.layernorm_embedding = LayerNorm(embed_dim)



            
        self.lm_head = self.build_lm_head(embed_dim=args.encoder_embed_dim,output_dim=len(dictionary),activation_fn=args.activation_fn,weight=(self.embed_tokens.weight)) 
        if self.mt:
            self.lm_head_left = self.build_lm_head(embed_dim=args.encoder_embed_dim,output_dim=len(dictionary),activation_fn=args.activation_fn,weight=(self.embed_tokens.weight)) 
            self.lm_head_right = self.build_lm_head(embed_dim=args.encoder_embed_dim,output_dim=len(dictionary),activation_fn=args.activation_fn,weight=(self.embed_tokens.weight)) 
           

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return SlstmLMHead(embed_dim, output_dim, activation_fn, weight)

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args,dictionary):
        return SLSTM(args,dictionary)

    def forward_embedding(self, src_tokens):
        x = embed = self.embed_tokens(src_tokens)
        if self.embed_positions is not None and not self.pos_h2g and not self.pos_g2h:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
       
        x, _ = self.forward_embedding(src_tokens)
        encoder_padding_mask = src_tokens.eq(self.padding_idx)     
        x, c, g, gc, _ = self.encoder(x, encoder_padding_mask, src_tokens)

        x_left = None
        x_right = None
        if not features_only:
            x_raw = self.output_layer(x, masked_tokens=masked_tokens)
            
            if self.mt:
                x_left = self.output_layer_left(x, masked_tokens=masked_tokens)
                x_right = self.output_layer_right(x, masked_tokens=masked_tokens)
            return x_raw,x_left,x_right, g
        # print(x.size())  #masksize V
        return x,None,None, g


    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens) 

    def output_layer_left(self, features, masked_tokens=None, **unused):
        if self.mt:
            return self.lm_head_left(features, masked_tokens) 
        else:
            return None

    def output_layer_right(self, features, masked_tokens=None, **unused):
        if self.mt:
            return self.lm_head_right(features, masked_tokens)
        else:
            return None 

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["encoder_out"]
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [encoder_out["encoder_padding_mask"][0].index_select(
                0, new_order
            )]

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class SLSTM(FairseqEncoder):
    def __init__(self, args, dictionary):
        super(SLSTM, self).__init__(dictionary)
        self.args = args
        
        self.num_layers = args.encoder_layers

        self.hidden_size = args.encoder_embed_dim
        
        self.padding_idx = dictionary.pad()# 1

        self.layer = self.build_encoder_layer(args)
        
    def build_encoder_layer(self, args):
        layer = SLSTM_block(args, self.padding_idx)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(self, x, mask, src_tokens):
        sequence_mask = torch.unsqueeze(1 - mask.type(torch.cuda.HalfTensor), dim=2)  
        masked_word_inputs = x * sequence_mask  # [bsz, seq_len, H]

        x_h, x_c, dummy_g_h, dummy_g_c = self.layer(masked_word_inputs, mask, src_tokens)
            
        return x_h, x_c, dummy_g_h, dummy_g_c ,None


class SLSTM_block(nn.Module):

    def __init__(self, args, padding_idx):
        super().__init__()
        self.slstm_kernel_size = args.slstm_kernel_size
        self.hidden_size = args.encoder_embed_dim

        self.num_layers = args.encoder_layers

        self.pos_g2h = args.pos_g2h
        self.pos_h2g = args.pos_h2g
        # self.use_layer_norm = args.use_layer_norm

        self.padding_idx = padding_idx
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )

        self.WV_t_ilrfsou = nn.Linear(4 * self.hidden_size, 7 * self.hidden_size, bias=True)
        self.U_t_ilrfsou = nn.Linear(1 * self.hidden_size, 7 * self.hidden_size, bias=False)
        self.WU_g_fo = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.WU_g_f_ = nn.Linear(2 * self.hidden_size, 1 * self.hidden_size, bias=True)
        
        self.ln=args.ln

        if self.ln:
            self.i_norm = LayerNorm(self.hidden_size)
            self.l_norm = LayerNorm(self.hidden_size)
            self.r_norm = LayerNorm(self.hidden_size)
            self.f_norm = LayerNorm(self.hidden_size)
            self.s_norm = LayerNorm(self.hidden_size)
            self.o_norm = LayerNorm(self.hidden_size)
            self.u_norm = LayerNorm(self.hidden_size)

            self.gf_norm = LayerNorm(self.hidden_size)
            self.go_norm = LayerNorm(self.hidden_size)
            self.gf__norm = LayerNorm(self.hidden_size)

        if self.pos_g2h or self.pos_h2g:
            self.embed_positions = (
                PositionalEmbedding(
                    args.max_source_positions,
                    args.encoder_embed_dim,
                    self.padding_idx,
                    learned=args.encoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )
        
        self.initgaussian = args.initgaussian
        self.reset_parameters()

    def reset_parameters(self):
        if self.initgaussian:
            print('initgaussian')
            nn.init.normal_(self.WV_t_ilrfsou.weight, mean=0, std=0.1)
            nn.init.normal_(self.U_t_ilrfsou.weight, mean=0, std=0.1)
            nn.init.normal_(self.WU_g_fo.weight, mean=0, std=0.1)
            nn.init.normal_(self.WU_g_f_.weight, mean=0, std=0.1) 
        else:
            nn.init.xavier_uniform_(self.WV_t_ilrfsou.weight)
            nn.init.xavier_uniform_(self.U_t_ilrfsou.weight)
            nn.init.xavier_uniform_(self.WU_g_fo.weight)
            nn.init.xavier_uniform_(self.WU_g_f_.weight)  


    def create_padding_variable(self, *shape):
        if torch.cuda.is_available():
            data = torch.zeros(*shape).to(device=torch.cuda.current_device())
        else:
            data = torch.zeros(*shape)
        var = autograd.Variable(data, requires_grad=False)
        return var

    def get_h_before(self, padding, hidden_states, step):
        _, src_len, _ = hidden_states.size()
        if step < src_len:
            displaced_hidden_states = hidden_states[:, :-step, :]
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return padding[:, :src_len, :]

    def get_h_after(self, padding, hidden_states, step):
        _, src_len, _ = hidden_states.size()
        if step < src_len:
            displaced_hidden_states = hidden_states[:, step:, :]
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return padding[:, :src_len, :]

    def forward(self, word_h, mask, src_tokens):
        shape = word_h.size()
        word_x = word_h.view(-1, shape[-1])
        Ux = self.U_t_ilrfsou(word_x)
        word_c = word_h
        # print(word_h.size())
        sequence_mask = torch.unsqueeze(1 - mask.type(torch.cuda.HalfTensor), dim=2)  #1.0     32,420,1
        sequence_lengths = torch.sum(sequence_mask, dim=1)# len1,...len_bsz  32,1

        dummy_g_h = torch.sum(word_h, dim=1) / sequence_lengths  # [bsz, H]
        dummy_g_c = torch.sum(word_c, dim=1) / sequence_lengths  # [bsz, H]

        batch_size, src_len, hidden_size = shape[0], shape[1], shape[2]

        padding_list = [
            self.create_padding_variable((batch_size, step + 1, hidden_size)).type(torch.cuda.HalfTensor)
            for step in range(self.slstm_kernel_size)
        ]

        mask_softmax_score = -mask.float() * 1e25  # 0.0  #32,420
        mask_softmax_score_expanded = torch.unsqueeze(mask_softmax_score, dim=2).type(torch.cuda.HalfTensor)  #32,420,1

        for layer_idx in range(self.num_layers):
            # print(word_h.size())
            word_h_before = [(self.get_h_before(padding_list[step], word_h, step + 1)* sequence_mask).view(-1, hidden_size) for step in range(self.slstm_kernel_size)]
            word_h_before = sum(word_h_before)

            word_h_after = [(self.get_h_after(padding_list[step], word_h, step + 1)* sequence_mask).view(-1, hidden_size) for step in range(self.slstm_kernel_size)]
            word_h_after = sum(word_h_after)
            # print(word_c.size())
            word_c_before = [(self.get_h_before(padding_list[step], word_c, step + 1)* sequence_mask).view(-1, hidden_size) for step in range(self.slstm_kernel_size)]
            word_c_before = sum(word_c_before)

            word_c_after = [(self.get_h_after(padding_list[step], word_c, step + 1)* sequence_mask).view(-1, hidden_size) for step in range(self.slstm_kernel_size)]
            word_c_after = sum(word_c_after)

            word_h_before_after = torch.cat([word_h_before, word_h_after], dim=1)

            word_h = word_h.view(-1, hidden_size)
            word_c = word_c.view(-1, hidden_size)


            dummy_g_h_expand = torch.unsqueeze(dummy_g_h, dim=1).repeat(1, src_len, 1)
            masked_dummy_g_h_expand = (dummy_g_h_expand * sequence_mask).view(-1, hidden_size)  

            dummy_g_c_expand = torch.unsqueeze(dummy_g_c, dim=1).repeat(1, src_len, 1).view(-1, hidden_size) #bsz seqlen, hidden

            all_gates = (
                self.WV_t_ilrfsou(torch.cat([word_h, word_h_before_after, masked_dummy_g_h_expand], dim=1))
                + Ux
            )
            v_i, v_l, v_r, v_f, v_s, v_o, v_u = torch.chunk(all_gates, 7, dim=1)

            if self.ln:
                v_i = self.i_norm(v_i)
                v_l = self.l_norm(v_l)
                v_r = self.r_norm(v_r)
                v_f = self.f_norm(v_f)
                v_s = self.s_norm(v_s)
                v_o = self.o_norm(v_o)
                v_u = self.u_norm(v_u)


            g_i = torch.sigmoid(v_i)
            g_l = torch.sigmoid(v_l)
            g_r = torch.sigmoid(v_r)
            g_f = torch.sigmoid(v_f)
            if not self.pos_g2h:
                g_s = torch.sigmoid(v_s)
            else:
                g_s = torch.sigmoid(v_s + self.embed_positions(src_tokens).view(-1, hidden_size))

            g_o = torch.sigmoid(v_o)
            g_u = torch.tanh(v_u)


            ilrfs_gates = torch.cat(
                    [g_i.unsqueeze(1), g_l.unsqueeze(1), g_r.unsqueeze(1), g_f.unsqueeze(1), g_s.unsqueeze(1)], dim=1
                )
            softmax_ilrfs_gates = F.softmax(ilrfs_gates, dim=1)
            g_i, g_l, g_r, g_f, g_s = torch.chunk(softmax_ilrfs_gates, 5, dim=1)
            g_i, g_l, g_r, g_f, g_s = g_i.squeeze(1), g_l.squeeze(1), g_r.squeeze(1), g_f.squeeze(1), g_s.squeeze(1)
            
            word_c = g_u * g_i + word_c_before * g_l + word_c_after * g_r + word_c * g_f + dummy_g_c_expand * g_s
            word_h = g_o * torch.tanh(word_c)


            word_h = self.dropout_module(word_h)

            word_h = word_h.view(batch_size, src_len, hidden_size)
            word_c = word_c.view(batch_size, src_len, hidden_size)
            word_h = word_h * sequence_mask
            word_c = word_c * sequence_mask

            ##########################################################################

            word_h_avg = torch.sum(word_h, dim=1) / sequence_lengths

            dummy_fo_gates = self.WU_g_fo(torch.cat([dummy_g_h, word_h_avg], dim=1))
            dummy_g_f, dummy_g_o = torch.chunk(dummy_fo_gates, 2, dim=1)

            dummy_g_f = torch.sigmoid(dummy_g_f)
            dummy_g_o = torch.sigmoid(dummy_g_o)

            if not self.pos_h2g:
                reshaped_hidden_output = word_h.view(-1, hidden_size)
            else:
                reshaped_hidden_output = (word_h + self.embed_positions(src_tokens)).view(-1, hidden_size)
            dummy_g_f__gates = self.WU_g_f_(                        
                    torch.cat([reshaped_hidden_output,masked_dummy_g_h_expand], dim=1)
                )

            if self.ln:
                dummy_g_f = self.gf_norm(dummy_g_f)
                dummy_g_o = self.go_norm(dummy_g_o)
                dummy_g_f__gates = self.gf__norm(dummy_g_f__gates)


            dummy_g_f_ = torch.sigmoid(dummy_g_f__gates)

            masked_dummy_g_f_ = (
                dummy_g_f_.view(shape[0], src_len, hidden_size) + mask_softmax_score_expanded
            )
            softmax_gates = F.softmax(torch.cat([masked_dummy_g_f_, torch.unsqueeze(dummy_g_f, dim=1)], dim=1),dim=1)
            

            g_f_ = softmax_gates[:, :src_len, :]
            g_g = softmax_gates[:, src_len:, :]      
            dummy_g_c = (
                torch.sum(g_f_ * word_c, dim=1)
                + torch.squeeze(g_g, dim=1) * dummy_g_c
            )

            dummy_g_h = dummy_g_o * torch.tanh(dummy_g_c)
            dummy_g_h = self.dropout_module(dummy_g_h)

        return word_h, word_c, dummy_g_h, dummy_g_c

@register_model_architecture("slstm", "slstm")
def base_architecture(args):

    args.dropout = getattr(args, "dropout", 0.1)
    args.max_source_positions = getattr(args, "max_positions", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.slstm_kernel_size = getattr(args, "slstm_kernel_size", 1)
    args.untie_weights = getattr(args, "untie_weights", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792) # 1024/1280/1536/1792/2048/2304
    args.pos_h2g = getattr(args, "pos_h2g", True)
    args.pos_g2h = getattr(args, "pos_g2h", True)
    args.use_layer_norm = getattr(args, "use_layer_norm", False)
    args.use_noise = getattr(args, "use_noise", False)
    args.mt = getattr(args, "mt", False)
    args.initgaussian = getattr(args, "initgaussian", False)
    args.ln = getattr(args, "ln", False)
    args.pos_input = getattr(args, "pos_input", False)

@register_model_architecture("slstm", "slstm1280")
def slstm1280_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1792")
def slstm1792_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    base_architecture(args)


@register_model_architecture("slstm", "slstm1792_pos_input")
def slstm1792_pos_input_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.pos_h2g = getattr(args, "pos_h2g", False)
    args.pos_g2h = getattr(args, "pos_g2h", False)
    args.pos_input = getattr(args, "pos_input", True)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1792_nopos")
def slstm1792_nopos_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.pos_h2g = getattr(args, "pos_h2g", False)
    args.pos_g2h = getattr(args, "pos_g2h", False)
    base_architecture(args)


@register_model_architecture("slstm", "slstm1792_ln")
def slstm1792_ln_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.ln = getattr(args, "ln", True)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1792_ln_posinput")
def slstm1792_ln_posinput_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.ln = getattr(args, "ln", True)
    args.pos_h2g = getattr(args, "pos_h2g", False)
    args.pos_g2h = getattr(args, "pos_g2h", False)
    args.pos_input = getattr(args, "pos_input", True)
    base_architecture(args)


@register_model_architecture("slstm", "slstm2048")
def slstm2048_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 2048)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1536_k3")
def slstm1536_k3_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.slstm_kernel_size = getattr(args, "slstm_kernel_size", 3)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1792_k3")
def slstm1792_k3_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.slstm_kernel_size = getattr(args, "slstm_kernel_size", 3)
    base_architecture(args)

@register_model_architecture("slstm", "slstm2304")
def slstm2304_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 2304)
    base_architecture(args)


@register_model_architecture("slstm", "slstm1280_mt")
def slstm1280_mt_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.mt = getattr(args, "mt", True)
    base_architecture(args)

@register_model_architecture("slstm", "slstm1792_mt")
def slstm1792_mt_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
    args.mt = getattr(args, "mt", True)
    base_architecture(args)

# @register_model_architecture("slstm", "slstm_pos")
# def slstm_pos_architecture(args):
#     args.pos_h2g = getattr(args, "pos_h2g", False)
#     args.pos_g2h = getattr(args, "pos_g2h", False)
#     base_architecture(args)

# @register_model_architecture("slstm", "slstm_nopos")
# def slstm_nopos_architecture(args):
#     args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", True)
#     args.pos_h2g = getattr(args, "pos_h2g", False)
#     args.pos_g2h = getattr(args, "pos_g2h", False)
#     base_architecture(args)


# @register_model_architecture("slstm", "slstm_g2h")
# def slstm_g2h_architecture(args):
#     args.pos_g2h = getattr(args, "pos_g2h", True)
#     base_architecture(args)

# @register_model_architecture("slstm", "slstm_1792_nonorm")
# def slstm_1792_nonorm_architecture(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
#     args.use_layer_norm = getattr(args, "use_layer_norm", False)
#     base_architecture(args)

# @register_model_architecture("slstm", "slstm_1792_nonorm_nonoise")
# def slstm_1792_nonorm_nonoise_architecture(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
#     args.use_layer_norm = getattr(args, "use_layer_norm", False)
#     args.use_noise = getattr(args, "use_noise", False)
#     base_architecture(args)

# @register_model_architecture("slstm", "slstm_1792_nonorm_10l")
# def slstm_1792_nonorm_10l_architecture(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1792)
#     args.use_layer_norm = getattr(args, "use_layer_norm", False)
#     args.encoder_layers = getattr(args, "encoder_layers", 10)
#     base_architecture(args)

