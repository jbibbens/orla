# Energy Per Token on Orla

- Energy per token is very dependent on entire stack
  - Decreases expo with batch size but levels off around 1024
  - Larger Models generally mean larger J/t
  - Prefill takes much more than decode, so output length amortizes high prefill cost
    - Similarly, longer context increases energy per token
  - Engine choice can make a big difference even with same model
    - Some engines appear more efficient for prefill vs decode

- TokenPowerBench repository contains outputs only for a subset of their experiments
  - There is code to allow you to do your own benchmarking

- Range
  - Low end is 0.02-0.03 J/token for edge models
  - Im having a hard time getting a sense of upper range: 10 J? 50J? 100 J? 200J?


- Proposed Experiment
  - fix 3-4 reasonable inference stacks and pick a J/token number which looks reasonable, probably something scaling: 0.5/2/8/32? Getting precise numbers will be hard without doing our own benchmarking
    - Is there a reason to consider anything time varying yet or not?
  - In the previous examples, Orla has expected a Prices output. Should we convert the energy values to prices or should orla take a simple energy usage signal separately?
