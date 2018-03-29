classdef MCAgent
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
        function agent = MCAgent(qFuncValues, ...
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
        
        function agent = policyEvaluation(agent, MDP, episode)
            agent.numUpdates = agent.numUpdates + 1;
            
            for episodeStep = episode
                state = episodeStep.state;
                
                % look for the first visit of this state
                firstVisitIdx = 0;
                
                for i = length(episode)
                    if isequal(episode(i).state, state)
                        firstVisitIdx = i;
                        break;
                    end
                end
                
                returnValue = 0;
                
                for i = firstVisitIdx:length(episode)
                    reward = episodeStep.reward;
                    returnValue = returnValue + (reward * agent.discountFactor^i);
                end

                % we update our value function using the return as the
                % target value. 
                % Target error = (return - current_estimate(s))
                stateFeatures = MDP.getStateFeatures(state);
                targetError = (returnValue - agent.valueFunction(stateFeatures));
                gradient = agent.valueFunctionGradient(stateFeatures);
                update = targetError*gradient;
                agent.valueParameters = ...
                    agent.valueParameters + agent.learningRate * update;
                
            end
           
        end
        
        function agent = policyControl(agent, MDP, episode)
            agent.numUpdates = agent.numUpdates + 1;
            
            for episodeStep = episode
                state = episodeStep.state;
                action = episodeStep.action;
                
                % look for the first visit of this state
                firstVisitIdx = 0;
                
                for i = length(episode)
                    if isequal(episode(i).state, state) && isequal(episode(i).action, action)
                        firstVisitIdx = i;
                        break;
                    end
                end
                
                returnValue = 0;
                
                for i = firstVisitIdx:length(episode)
                    reward = episodeStep.reward;
                    returnValue = returnValue + (reward * agent.discountFactor^i);
                end

                % we update our value function using the return as the
                % target value. 
                % Target error = (return - current_estimate(s))
                stateFeatures = MDP.getStateFeatures(state);
                targetError = (returnValue - agent.qValueFunction(stateFeatures, action));
                gradient = agent.valueFunctionGradient(stateFeatures);
                update = targetError*gradient;
                agent.qValueParameters(:, :, action) = ...
                    agent.qValueParameters(:, :, action) + agent.learningRate * update;
                
            end
           
        end
        
        function stateValue = valueFunction(agent, stateFeatures)
            stateValue = sum(sum(stateFeatures .* agent.valueParameters));
        end
        
        % Samples an action in an epsilon greedy manner
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

