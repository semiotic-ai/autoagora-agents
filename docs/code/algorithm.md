
``` mermaid
graph LR
    Algorithm --> PredeterminedAlgorithm
    Algorithm --> BanditAlgorithm
    BanditAlgorithm --> VPGBandit
    BanditAlgorithm --> PPOBandit
    PPOBandit --> RollingMemoryPPOBandit

``` 
::: autoagora_agents.algorithm