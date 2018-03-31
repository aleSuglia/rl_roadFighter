classdef TDAgent
    %TDAgent learning agent which exploits Q-Learning to learn to 
    % navigate in the environment
    properties
        qValueParameters
        valueParameters
        numActions
        learningRate
        initLearningRate
        numUpdates
        lambdaDecay
        discountFactor
    end
    
    methods
        function agent = TDAgent(qFuncValues, ...
                                 numActions, ...
                                 learningRate, ...
                                 discountFactor)
            agent.qValueParameters = qFuncValues;
            dims = size(qFuncValues);
            agent.valueParameters = rand(dims(1:2));
            agent.numActions = numActions;
            agent.learningRate = learningRate;
            agent.initLearningRate = learningRate;
            agent.numUpdates = 0;
            agent.lambdaDecay = 1.0;
            agent.discountFactor = discountFactor;
        end
        
        % state: state feature vector
        % action: action index
        function value = qValueFunction(agent, stateFeatures, action)
            value = sum ( sum( agent.qValueParameters(:,:,action) .* stateFeatures ) );
            
        end

        function value = valueFunction(agent, stateFeatures)
            value = sum ( sum( agent.valueParameters .* stateFeatures ) );
            
        end
        
        function agent = policyEvaluation(agent, MDP, episode)
            agent.numUpdates = agent.numUpdates + 1;
            
            episodeIdx = 1;
            for episodeStep = episode
                state = episodeStep.state;
                action = episodeStep.action;
                
                % we do a lookahead to the next state
                if episodeIdx+1 < length(episode)
                    nextState = episode(episodeIdx+1).state;
                    nextStateFeatures = MDP.getStateFeatures(nextState);
                    nextStateValue = agent.valueFunction(nextStateFeatures);
                else
                    % last step of the episode
                    nextStateValue = 0;
                end
                % we update our value function using the return as the
                % target value. 
                % Target error = ((r + Vestimate(next_state)) - Vestimate(s, a))
                stateFeatures = MDP.getStateFeatures(state);
                tdTarget = episodeStep.reward + nextStateValue;
                targetError = (tdTarget - agent.valueFunction(stateFeatures));
                gradient = agent.valueFunctionGradient(stateFeatures);
                update = targetError*gradient;
                agent.valueParameters(:, :) = ...
                    agent.valueParameters(:, :) + agent.learningRate * update;
                episodeIdx = episodeIdx + 1;
            end
        end
        
        function agent = policyControl(agent, MDP, episode)
            agent.numUpdates = agent.numUpdates + 1;
            
            episodeIdx = 1;
            for episodeStep = episode
                state = episodeStep.state;
                action = episodeStep.action;
                
                % we do a lookahead to the next state
                if episodeIdx+1 < length(episode)
                    actionValues = zeros(1, agent.numActions);
                    nextState = episode(episodeIdx+1).state;
                    nextStateFeatures = MDP.getStateFeatures(nextState);
                    for i = agent.numActions
                        actionValues(i) = agent.qValueFunction(nextStateFeatures, i);
                    end

                    [~, maxQValue] = max(actionValues);
                else
                    % last step of the episode
                    maxQValue = 0;
                end
                % we update our value function using the return as the
                % target value. 
                % Target error = ((r + maxQValue) - current_estimate(s, a))
                stateFeatures = MDP.getStateFeatures(state);
                tdTarget = episodeStep.reward + maxQValue;
                targetError = (tdTarget - agent.qValueFunction(stateFeatures, action));
                gradient = agent.valueFunctionGradient(stateFeatures);
                update = targetError*gradient;
                agent.qValueParameters(:, :, action) = ...
                    agent.qValueParameters(:, :, action) + agent.learningRate * update;
                episodeIdx = episodeIdx + 1;
            end
        end
        
        function action = predict(agent, stateFeatures, epsilon)
            % every action receives epsilon/agent.numActions scores
            actionProbs = ones(1, agent.numActions) * epsilon / agent.numActions;
            actionValues = zeros(1, agent.numActions);
            % evaluate Q-function for each action
            for action = 1:agent.numActions
                actionValues(action) = agent.qValueFunction(stateFeatures, action);
            end % for each possible action
            
            % the best action receives a weight boost
            [~, bestAction] = max(actionValues);
            actionProbs(bestAction) = actionProbs(bestAction) + (1.0 - epsilon);
            % consider the weight as a probability vector and samples an
            % action according to the defined probability weights
            action = randsample(1:agent.numActions, 1, true, actionProbs);
        end

        % assumption: the gradient for a linear function approximator 
        % is represented by the feature values
        function gradient = valueFunctionGradient(~, state)
            gradient = state;
        end
    end
end

