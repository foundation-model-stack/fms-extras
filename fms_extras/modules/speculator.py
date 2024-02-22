import torch
import torch.nn as nn
import torch.nn.functional as F
from fms.modules.layernorm import LayerNormParameterized

class Speculator(nn.Module):
    def __init__(self, emb_dim=4096, inner_dim=0, vocab_size=32000, n_predict=3):
        super().__init__()
        self.npredict = n_predict
        self.emb_dim = emb_dim
        inner_dim = inner_dim if inner_dim!=0 else emb_dim
        self.inner_dim = inner_dim
        self.vsize = vocab_size
        self.emb = nn.ModuleList([nn.Embedding(vocab_size, inner_dim) for _ in range(n_predict)])
        self.proj = nn.ModuleList([nn.Linear((emb_dim if i==0 else inner_dim), inner_dim, bias=False) for i in range(n_predict)])
        self.head = nn.ModuleList([nn.Linear(inner_dim, vocab_size, bias=False) for _ in range(n_predict)])
        self.ln = nn.ModuleList(
            [LayerNormParameterized(inner_dim, elementwise_shift=True, elementwise_scale=True) for _ in range(n_predict)]
        )
        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = .5**(.5/n_predict)
        self.emb_weight = (1-self.state_weight**2)**.5
        self.a = nn.GELU()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 1 / self.inner_dim**0.5)
            elif isinstance(m, LayerNormParameterized):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def generate_suffixes(self, state, ind, topk=[5, 4, 3], n=5):
        """
        FOR INFERENCE
        ----
        Generate tree of candidate sequences given latest base model embedding (state) and chosen token (ind).
        Topk indicates # of tree "branches" at each head.
        n pares down the candidate list from prod(topk) to the top n most confident.
        """
        # state: b 1 d
        # ind: b 1
        # k indicates # of candidates
        # h indicates # of generated tokens
        b = state.size(0)
        out = torch.empty(b, 1, 0, device=state.device).int()  # b k h
        log_probs = torch.zeros(b, 1, device=state.device)  # b k
        assert (
            len(topk) == self.n_predict
        ), f"You must provide a topk number for each head ({self.n_predict} heads, {len(topk)} provided)"
        for i in range(self.n_predict):
            # Project and predict
            z = self.emb[i](ind).mul(self.emb_weight*(self.inner_dim/2)**.5)  # b k d
            state = self.a(self.ln[i](self.proj[i](state)*self.state_weight+z))  # b k d
            probs = F.log_softmax(self.head[i](state), dim=2)  # b k v
            probs, preds = probs.topk(topk[i], dim=2)  # b k k'

            # Update candidate set with new predictions
            out = out.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' h
            out = torch.cat([out, preds.unsqueeze(3)], dim=3)  # b k k' h+1
            out = out.view(b, -1, i + 1)  # b kk' h+1

            # Update state, log_probs and ind for new predictions
            state = state.unsqueeze(2).expand(-1, -1, topk[i], -1)  # b k k' d
            state = state.reshape(b, -1, state.size(3))  # b kk' d
            ind = preds.view(b, -1)  # b kk'
            log_probs = log_probs.unsqueeze(2).expand(b, -1, topk[i])  # b k k'
            log_probs = log_probs.add(probs).reshape(b, -1)  # b kk'

        # Take only top n best guesses
        best_guesses = log_probs.topk(n, dim=1)[1]  # b k
        return out.gather(
            1, best_guesses.unsqueeze(2).expand(-1, -1, self.n_predict)
        )  # b n h

    def forward(self, state, inds):
        """
        FOR TRAINING
        ----
        Since we're assuming all prior tokens are "correct", don't act recursively, just pull from provided inds.
        Produces self.n_predict predicted tokens for each token embedding in state.
        Inds requires self.n_predict extra tokens on the right to "simulate" recursive behavior for end positions.
        """
        # state: b n d
        # inds: b n+h (..., pred token, n+2, n+3, n+4)
        out = []
        for i in range(self.npredict):
            z = self.emb[i](inds[:, i : i + state.size(1)]).mul(self.emb_weight*(self.inner_dim/2)**.5)  # b n d
            state = self.a(self.ln[i](self.proj[i](state)*self.state_weight+z))  # b n d
            out.append(self.head[i](state))  # b n v
        return torch.stack(out, dim=0)  # h b n v