"""
An *attention* module that computes the similarity between
an input vector and the rows of a matrix.
"""

import torch

from overrides import overrides
from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax


def replace_indices_with_tokens_from_vocab(tensor, vocab_dict):
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        tensor = tensor.cpu().numpy()
    return [[vocab_dict[key].replace("#", "") for key in row] for row in tensor]

def replace_indices_with_tokens_from_vocab_1d(tensor, vocab_dict):
    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        tensor = tensor.cpu().numpy()
    return [[vocab_dict[key].replace("#", "") for key in tensor]]

#re-assemble BERT-wordpieces to full tokens
def unwordpiece_bert_tokens(tensor, offsets2d, vocab):
    res = []
    for x, offsets in enumerate(offsets2d.cpu().numpy()):
        subres = []
        for i, offset in enumerate(offsets):
            subsubres = []
            if i != len(offsets) - 1 and offset != 0:
                if offsets[i + 1] == offset + 1:
                    subsubres.append([tensor[x, offset].item()])
                else:
                    subsubres.append(tensor[x, offset : offsets[i + 1]].tolist())
                i += 1
            else:
                subsubres.append([tensor[x, offset].item()])
                subres.append(subsubres)
                break
            subres.append(subsubres)
        stringRes = []
        for r in subres:
            stringRes.append("".join(replace_indices_with_tokens_from_vocab(r, vocab)[0]))
        res.append(stringRes)
    return res



class Attention(torch.nn.Module, Registrable):
    """
    An ``Attention`` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.


    Inputs:

    - vector: shape ``(batch_size, embedding_dim)``
    - matrix: shape ``(batch_size, num_rows, embedding_dim)``
    - matrix_mask: shape ``(batch_size, num_rows)``, specifying which rows are just padding.

    Output:

    - attention: shape ``(batch_size, num_rows)``.

    Parameters
    ----------
    normalize : ``bool``, optional (default: ``True``)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """
    def __init__(self,
                 normalize: bool = True) -> None:
        super().__init__()
        self._normalize = normalize

    @overrides
    def forward(self,
                vector: torch.Tensor,
                matrix: torch.Tensor,
                matrix_mask: torch.Tensor = None,
                source_tokens = None, target_tokens = None, vocab_cust = None, last_prediction = None) -> torch.Tensor:



        similarities = self._forward_internal(vector, matrix)
        if self._normalize:

            softmax_sims = masked_softmax(similarities, matrix_mask)
            if source_tokens is None or target_tokens is None or vocab_cust is None:
                return softmax_sims
            else:
                inputSeq = unwordpiece_bert_tokens(source_tokens['bert'], source_tokens['bert-offsets'], vocab_cust['bert'])
                predTokens = replace_indices_with_tokens_from_vocab_1d(last_prediction, vocab_cust['target'])[0]
                attention = (inputSeq, predTokens, softmax_sims.cpu())

                return softmax_sims, attention
        else:
            return similarities

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
