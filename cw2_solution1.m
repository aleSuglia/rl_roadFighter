

%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;


%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

noCarOnRowProbability = 0.8 ; % the probability that there is no car 
% spawned for each row

seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );


%% Initialising the state observation (state features) and setting up the 
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)

% Monte-Carlo: 0
% TD-learning (Q-Learning): 1
ALGORITHM = 0;
originalEpsilon = 1;
epsilon = originalEpsilon;
lambda = 1.0;
numTrainEpisodes = 1000;

if ALGORITHM == 0
    agent = MCAgent(...
        Q_test1, ...
        3, ... % numActions
        0.01, ... % learningRate
        1 ... % discountFactor
    );
else
    agent = TDAgent(...
        Q_test1, ...
        3, ... % numActions
        0.01, ... % learningRate
        1 ... % discountFactor
    );
end
    
clear episode;
prevValueParameters = zeros(4,5);

for e = 1:numTrainEpisodes
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
        blockSize, noCarOnRowProbability, ...
        probabilityOfUniformlyRandomDirectionTaken, rewards );
    currentMap = MDP ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
    
    % intialise the current episode data
    for i = episodeLength
        episode(i).state = [0, 0];
        episode(i).action = 0;
        episode(i).reward = 0;
    end
    % exploring starts
    currState = [randi(MDP.GridSize(1)), randi(MDP.GridSize(2))];
    % If you need to keep track of agent movement history:
    currentTimeStep = 0;
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep+1, :) = currState ;

    % The agent interacts with the environment and experiences
    % a new trajectory
    for i = 1:episodeLength
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(currState); % dimensions are 4rows x 5columns
        % act greedily with respect to the Q_test1 policy
        for action = 1:3
            action_values(action) = ...
                sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
        end % for each possible action
        [~, actionTaken] = max(action_values);
        
        % The agent is in the state 'currState' and executes the
        % action 'actionTaken'
        [ agentRewardSignal, nextState, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( actionTaken, currState, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;

        % Save training data (state, action, reward)
        episode(i).state = currState;
        episode(i).action = actionTaken;
        episode(i).reward = agentRewardSignal;

        currState = nextState;
    end
    
    % improve the value function parameters 
    agent = agent.policyEvaluation(MDP, episode);
    disp("V-function parameters after Gradient descent step");
    disp(agent.valueParameters);
    
    % decay the learning rate and the epsilon every 5 episodes
    if mod(e, 5) == 0 
        agent.learningRate = agent.initLearningRate * exp(-agent.lambdaDecay*agent.numUpdates);
    end
    
    if abs(agent.valueParameters - prevValueParameters) < eps(0.5)
        disp("V-function parameters are not changing anymore.")
        break
    end
    
    prevValueParameters = agent.valueParameters;
end % for each episode

disp("Number of training episodes completed " + e + " over " + numTrainEpisodes); 


