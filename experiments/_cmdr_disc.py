"""DiscreteRescor smoke test."""
import sys, torch
sys.path.insert(0, 'src')
from wmca.modules.discrete_rescor import DiscreteRescor

dr = DiscreteRescor(vocab_size=256, n_actions=17, embed_dim=64)
pc = dr.param_count()
tokens = torch.randint(0, 256, (2, 16, 16))
actions = torch.randint(0, 17, (2,))
logits = dr(tokens, actions)
loss = torch.nn.CrossEntropyLoss()(logits, torch.randint(0, 256, (2, 16, 16)))
print(f'DiscreteRescor: trained={pc["trained"]:,} frozen={pc["frozen"]:,}')
print(f'Forward: tokens={list(tokens.shape)} -> logits={list(logits.shape)}')
print(f'CE loss: {loss.item():.4f}')
print('SMOKE TEST PASSED')
