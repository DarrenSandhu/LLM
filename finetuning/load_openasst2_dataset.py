from datasets import load_dataset, load_from_disk
import os
# Load OpenAssistant dataset
if not os.path.exists('/Users/darrensandhu/Projects/LLM/openassistant/oasst2'):
    openassistant = load_dataset('OpenAssistant/oasst2')
    train = openassistant['train']  # len(train) = 128575 (95%)
    val = openassistant['validation']  # len(val) = 6599 (5%)
    openassistant.save_to_disk('OpenAssistant/oasst2')
else:
    openassistant = load_dataset('OpenAssistant/oasst2')
    train = openassistant['train']
    print(f"Training samples: {len(train)}")
    val = openassistant['validation']
    print(f"Validation samples: {len(val)}")
