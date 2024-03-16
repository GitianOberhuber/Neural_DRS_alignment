from typing import Dict, Optional

import torch
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_decoders.seq_decoder import SeqDecoder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.common.util import fixed_seeds

@Model.register("composed_seq2seq")
class ComposedSeq2Seq(Model):
    """
    This ``ComposedSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    The ``ComposedSeq2Seq`` class composes separate ``Seq2SeqEncoder`` and ``SeqDecoder`` classes.
    These parts are customizable and are independent from each other.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_text_embedders : ``TextFieldEmbedder``, required
        Embedders for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    decoder : ``SeqDecoder``, required
        The decoder of the "encoder/decoder" model
    tied_source_embedder_key : ``str``, optional (default=``None``)
        If specified, this key is used to obtain token_embedder in `source_text_embedder` and
        the weights are shared/tied with the decoder's target embedding weights.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_text_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: SeqDecoder,
                 tied_source_embedder_key: Optional[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 encoder_feat: Optional[Seq2SeqEncoder] = None,
                 char_text_embedder: TextFieldEmbedder = None)-> None:

        super(ComposedSeq2Seq, self).__init__(vocab, regularizer)
        fixed_seeds()
        self._source_text_embedder = source_text_embedder
        self._char_text_embedder = char_text_embedder
        self._encoder = encoder
        self._decoder = decoder
        self._encoder_feat = encoder_feat

        if tied_source_embedder_key:
            # A bit of a ugly hack to tie embeddings.
            # Works only for `BasicTextFieldEmbedder`, and since
            # it can have multiple embedders, and `SeqDecoder` contains only a single embedder, we need
            # the key to select the source embedder to replace it with the target embedder from the decoder.
            if not isinstance(self._source_text_embedder, BasicTextFieldEmbedder):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Source text embedder is not an instance of `BasicTextFieldEmbedder`.")
            # pylint: disable=protected-access
            source_embedder = self._source_text_embedder._token_embedders[tied_source_embedder_key]
            if not isinstance(source_embedder, Embedding):
                raise ConfigurationError("Unable to tie embeddings,"
                                         "Selected source embedder is not an instance of `Embedding`.")
            if source_embedder.get_output_dim() != self._decoder.target_embedder.get_output_dim():
                raise ConfigurationError(f"Output Dimensions mismatch between"
                                         f"source embedder and target embedder.")
            self._source_text_embedder._token_embedders[tied_source_embedder_key] = self._decoder.target_embedder
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                char_tokens: Dict[str, torch.LongTensor] = None,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass on the encoder and decoder for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        Returns
        -------
        Dict[str, torch.Tensor]
            The output tensors from the decoder.
        """
        # Make sure that we encode the features separately if we do two encoders

        final_state = {}
        keep_enc = {}
        for key in source_tokens:
            # Don't do separate feature encoding for two-encoders if we also have characters
            # Characters are only added individually in a single encoder (two encoders max)
            if (not char_tokens) and key in ['ccg', 'pos', 'dep', 'lem', 'sem', 'lem_glove'] and self._encoder_feat:
                state = self._encode({key : source_tokens[key]}, ident=key)
                final_state.update(state)
            else:
                keep_enc[key] = source_tokens[key]
        # Check if we do character-encoding
        if char_tokens:
            state = self._encode(char_tokens, ident="char")
            final_state.update(state)

        # Finally do token encoding
        tok_state = self._encode(keep_enc)
        final_state.update(tok_state)

        stuff = self._decoder(final_state, target_tokens, self.vocab._index_to_token, source_tokens)
        return stuff

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        return self._decoder.post_process(output_dict)

    def _encode(self, source_tokens: Dict[str, torch.Tensor], ident: str = '') -> Dict[str, torch.Tensor]:
        """
        Make foward pass on the encoder.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.

        Returns
        -------
        Dict[str, torch.Tensor]
            Map consisting of the key `source_mask` with the mask over the
            `source_tokens` text field,
            and the key `encoder_outputs` with the output tensor from
            forward pass on the encoder.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        if ident == "char":
            embedded_input = self._char_text_embedder(source_tokens)
        else:
            embedded_input = self._source_text_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        # Use feature encoder for features
        if ident in ['ccg', 'pos', 'dep', 'lem', 'sem', 'lem_glove']:
            encoder_outputs = self._encoder_feat(embedded_input, source_mask)
        elif ident == "char":
            encoder_outputs = self._encoder_feat(embedded_input, source_mask)
        else:
            encoder_outputs = self._encoder(embedded_input, source_mask)
        return {
                "source_mask" + ident: source_mask,
                "encoder_outputs" + ident: encoder_outputs,
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._decoder.get_metrics(reset)
