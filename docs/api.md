# Python API

## Agents

::: autoagora_agents.agent

::: autoagora_agents.agent_factory
``` mermaid
graph LR
    Action:::build --> AgentFactory
    Optimizer:::build --> AgentFactory
    Policy:::build --> AgentFactory
    AgentFactory:::build --> Agent:::build

    classDef build fill:#f96

``` 


::: autoagora_agents.action_mixins
``` mermaid
graph LR
    Action --> DeterministicAction
    Action --> GaussianAction
    Action --> ScaledGaussianAction
```

::: autoagora_agents.policy_mixins
``` mermaid
graph LR
    Policy --> NoUpdatePolicy
    Policy --> ExperienceBufferPolicy
    ExperienceBufferPolicy --> ProximalPolicyOptimization
    ProximalPolicyOptimization --> RollingMemoryPPO
    ExperienceBufferPolicy --> VanillaPolicyGradient
```

::: autoagora_agents.optimizer_mixins

::: autoagora_agents.reinforcement_learning_policy_mixins

## Enviroments
``` mermaid
graph LR
    Environment --> SimulatedSubgraph
    SimulatedSubgraph --> NoisySharedSubgraph
    NoisySharedSubgraph --> NoisyCyclicSharedSubgraph
    SimulatedSubgraph --> NoisySimulatedSubgraph
    NoisySimulatedSubgraph --> NoisyQueriesSubgraph
    NoisySimulatedSubgraph --> NoisyDynamicQueriesSubgraph
    NoisySimulatedSubgraph --> NoisyCyclicZeroQueriesSubgraph
    NoisySimulatedSubgraph --> NoisyCyclicQueriesSubgraph
```

::: environments.environment

``` mermaid
graph LR
    EnvironmentFactory:::build -->  Environment:::build

    classDef build fill:#f96
```
::: environments.environment_factory

::: environments.shared_subgraph

::: environments.simulated_subgraph
