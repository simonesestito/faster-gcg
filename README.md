# Faster-GCG
Implementation of Faster-GCG Algorithm (Li et al., 2024)

# Usage
```sh
git clone https://github.com/simonesestito/faster-gcg
pip install ./faster-gcg/
```

You can refer to the following example Python code:
```python
import gcg
from transformers import PreTrainedModel, PreTrainedTokenizerBase

model: PreTrainedModel = ...
tokenizer: PreTrainedTokenizerBase = ...

attacker = gcg.FasterGCG(
      num_iterations=10_000,
      batch_size=900,
      adversarial_tokens_length=15,
      top_k_substitutions_length=64,
      vocab_size=tokenizer.vocab_size,
      lambda_reg_embeddings_distance=0.1,
)

target_response = ' and it was a sunny day.'

x_suffix_ids, y_attack_response_ids, x_attack_str, y_response_str, steps = \
    attacker.tokenize_and_attack(tokenizer,
                                 model,
                                 x_fixed_input=None,
                                 y_target_output=target_response,
                                 show_progress=True)

print(f"Attack string: {x_attack_str}")
print(f"Attack response: {y_response_str}")
print(f"Desired response: {target_response}")
print(f"Steps: {steps}")
```


## License
Based on the algorithm originally published
in [Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models](https://arxiv.org/abs/2410.15362)
by Xiao Li, Zhuhong Li, Qiongxiu Li, Bingze Lee, Jinghao Cui, Xiaolin Hu


**Paper citation**
```bibtex
@misc{li2024fastergcgefficientdiscreteoptimization,
      title={Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models}, 
      author={Xiao Li and Zhuhong Li and Qiongxiu Li and Bingze Lee and Jinghao Cui and Xiaolin Hu},
      year={2024},
      eprint={2410.15362},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.15362}, 
}
```

**Implementation license**

    Faster-GCG Implementation
    Copyright (C) 2025  Simone Sestito

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
