classdef Agent
    %AGENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        stateValues
        policy
        location
        numActions
    end
    
    methods
        function agent = Agent(MDP, policy, agentLocation, numActions)
            %AGENT Construct an instance of this class
            %   Receives the gridMap instance so that it is possible 
            %   to initialise the state value tensor (width, height,
            %   num_actions)
            agent.stateValues = zeros(MDP.GridSize(1), MDP.GridSize(2));
            agent.policy = policy;
            agent.location = agentLocation;
            agent.numActions = numActions;
        end
        
        function agent = updateLocation(agent, newAgentLocation)
            agent.location = newAgentLocation;
        end
        
        function [agent, numIterations] = policyEvaluation(agent, MDP, epsilon)
           stopCondition = false;
           numIterations = 0; 
           
            % policy evaluation algorithm
            while (stopCondition == false)
                maxDiff = 0;
                numIterations = numIterations + 1;
                
                for i = (1:MDP.GridSize(1))
                    for j = (1:MDP.GridSize(2))
                        currState = [i,j];
                        % from the given state (i,j) evaluate 
                        % the value function 
                        currStateValue = 0;
                        prevStateValue = agent.stateValues(i, j);
                        
                        % N.B.: 
                        % the current policy is deterministic: the probability
                        % associated to the selected action is always 1
                        % 0 for all the other!
                        action = agent.policy(i, j);
                        
                        [nextStates, probs] = MDP.getTransitions(...
                            currState, ...
                            action ...
                        );
                        nextStatesSize = size(nextStates);
                        for ns = 1:nextStatesSize(1)
                            nextState = nextStates(ns, :);
                            reward = MDP.getReward(currState, nextState, action);
                            % we can omit the action probability because it
                            % will be always 1 for the selected action, 
                            % 0 otherwise
                            currStateValue = currStateValue + probs(ns) * (reward + agent.stateValues(nextState(1), nextState(2)));
                        end
                        
                        % We don't have two copies of the value function
                        % matrix because we replace the old value in-place.
                        % Despite we may end up using new values and old
                        % values of the value function for some states,
                        % this method has been demonstrated to converge to
                        % an optimal policy as well.
                        agent.stateValues(i, j) = currStateValue;
                        
                        diff = abs(agent.stateValues(i,j) - prevStateValue);
                        
                        if (diff > maxDiff)
                            maxDiff = diff;
                        end
                    end
                end
                
                stopCondition = maxDiff < epsilon;
            end
 
        end
        
        function [agent, policyStable] = policyImprovement(agent, MDP)
            policyStable = true;
     
            for i = (1:MDP.GridSize(1))
                for j = (1:MDP.GridSize(2))
                    currState = [i,j];
                    bestActionValueFunction = 0;
                    actionTaken = agent.policy(i, j);
                    bestAction = actionTaken;
                    
                    for a = (1:agent.numActions)
                        [nextStates, probs] = MDP.getTransitions(...
                                currState, ...
                                a ...
                         );


                        currActionStateValue = 0;

                        nextStatesSize = size(nextStates);
                        for ns = 1:nextStatesSize(1)
                            nextState = nextStates(ns, :);
                            reward = MDP.getReward(currState, nextState, a);
                            currActionStateValue = currActionStateValue + probs(ns) * (reward + agent.stateValues(nextState(1), nextState(2)));
                        end

                        if (currActionStateValue > bestActionValueFunction)
                            bestActionValueFunction = currActionStateValue;
                            bestAction = a;
                        end
                    end
                    
                    agent.policy(currState(1), currState(2)) = bestAction;
                    
                    if (bestAction ~= actionTaken)
                        policyStable = false;
                    end
                end
            end
        end
        
        function action = act(agent)
            action = agent.policy(agent.location(1), agent.location(2));
        end
        
    end
end

