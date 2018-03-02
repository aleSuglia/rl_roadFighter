classdef Agent
    %AGENT Class that represents a generic agent that learns to solve
    % a given MDP
    % The class represents an agent with the following attributes:
    % - stateValues: a matrix whose cells contains the value function for
    % each state
    % - policy: a matrix whose cells contains the action to take for each
    % state (the policy is deterministic)
    % - location: the position of the agent in the MDP
    % - numActions: number of actions that the agent can execute in the
    % environment
    
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
        
        % Updates the location of the agent by using the one specified as a
        % parameter
        function agent = updateLocation(agent, newAgentLocation)
            agent.location = newAgentLocation;
        end
        
        % Executes the policy evaluation algorithm for the specified agent
        % that interacts with the environment whose transition dynamics are
        % specified by the MDP. The algorithm is executed until the 
        % difference between value function at the step k and k+1 is less
        % than epsilon
        % 
        % Returns:
        %   - agent: updated agent instance
        %   - numIterations: number of iterations required for the
        %   algorithm to converge to an "optimal" policy.
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
                        
                        action = agent.policy(i, j);
                        
                        [nextStates, probs] = MDP.getTransitions(...
                            currState, ...
                            action ...
                        );
                        nextStatesSize = size(nextStates);
                        for ns = 1:nextStatesSize(1)
                            nextState = nextStates(ns, :);
                            reward = MDP.getReward(currState, nextState, action);
                            % N.B.: 
                            % the current policy is deterministic: the probability
                            % associated to the selected action is always 1
                            % 0 for all the others!
                            currStateValue = currStateValue + probs(ns) * (reward + agent.stateValues(nextState(1), nextState(2)));
                        end
                        
                        % We don't have two copies of the value function
                        % matrix because we replace the old value in-place.
                        % Despite we may end up using new values and old
                        % values of the value function for some states,
                        % this method has been demonstrated to converge to
                        % an optimal policy as well. This algorithm
                        % converges faster than the classical version
                        % because it exploits the new value function as
                        % soon as they are evaluated.
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
        
        % Executes the policy improvement algorithm in order to improve the
        % current agent policy according to its value function and the MDP
        % specified.
        % 
        % Returns:
        %   - agent: agent instance with the improved policy
        %   - policyStable: True if the policy is stable, False otherwise
        function [agent, policyStable] = policyImprovement(agent, MDP)
            policyStable = true;
     
            for i = (1:MDP.GridSize(1))
                for j = (1:MDP.GridSize(2))
                    currState = [i,j];
                    % temporary vector to store the value associated to
                    % each action
                    actionValues = zeros(1, agent.numActions);
                    % current best action taken by the policy
                    actionTaken = agent.policy(i, j);
           
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
                        
                        actionValues(a) = currActionStateValue;
                    end
                    
                    % determine the best action according to the newly 
                    % evaluated value functions
                    [~, bestAction] = max(actionValues);
                    
                    % Improve the policy by setting the current bestAction
                    % as the action that should be taken for the current
                    % state. We act greedy with respect to the value
                    % function
                    agent.policy(currState(1), currState(2)) = bestAction;
                    
                    if (bestAction ~= actionTaken)
                        policyStable = false;
                    end
                end
            end
        end
       
        % Selects the action that the agent should execute in the current
        % state.
        %
        % Returns:
        %   - action: index of the selected action
        function action = act(agent, state)
            action = agent.policy(state(1), state(2));
        end
        
    end
end

