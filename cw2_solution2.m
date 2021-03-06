

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
    rewards);


%% Initialising the state observation (state features) and setting up the 
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)


% Monte-Carlo: 0
% TD-learning (Q-Learning): 1
ALGORITHM = 1;
% probability to take a random action in the epsilon greedy strategy
originalEpsilon = 0.8;
epsilon = originalEpsilon;
% decay rate for the epsilon greedy strategy
lambda = 0.9;
numTestEpisodes = 10;
numTrainEpisodes = 1000;

if ALGORITHM == 0
    agent = MCAgent(...
        Q_test1, ...
        3, ... % numActions
        0.01, ... % learningRate
        1 ... % discountFactor, 
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
prevQParameters = agent.qValueParameters;

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
    %currState = [randi(MDP.GridSize(1)), randi(MDP.GridSize(2))];
    % If you need to keep track of agent movement history:
    currState = MDP.Start;
    currentTimeStep = 0;
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep+1, :) = currState ;

    % The agent interacts with the environment and experiences
    % a new trajectory
    for i = 1:episodeLength
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(currState); % dimensions are 4rows x 5columns
        actionTaken = agent.predict(stateFeatures, epsilon);
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
    
    agent = agent.policyControl(MDP, episode);
    disp("Q-function parameters after Gradient descent step");
    for a = (1:agent.numActions)
        disp("----- action " + a + " -----")
        disp(agent.qValueParameters(:, :, a));
    end
    
    % decay the learning rate and the epsilon every 5 episodes
    if mod(e, 5) == 0 
        agent.learningRate = agent.initLearningRate * exp(-agent.lambdaDecay*agent.numUpdates);
        epsilon = originalEpsilon * exp(-lambda*e);
    end
    
    if abs(agent.qValueParameters - prevQParameters) < eps(0.5)
        disp("Q-function parameters are not changing anymore.")
        break
    end
end % for each episode

disp("Starting evaluation step");
testReturns = zeros(1, numTestEpisodes);

for e = 1:numTestEpisodes
    MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
    blockSize, noCarOnRowProbability, ...
    probabilityOfUniformlyRandomDirectionTaken, rewards );
    currentMap = MDP ;
    %%
    currentTimeStep = 0 ;
    agentLocation = currentMap.Start ;
    startingLocation = agentLocation ; % Keeping record of initial location.
    
    % If you need to keep track of agent movement history:
    agentMovementHistory = zeros(episodeLength+1, 2) ;
    agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;
        
    realAgentLocation = agentLocation ; % The location on the full test map.
    Return = 0;
    
    for i = 1:episodeLength
        
        % Use the $getStateFeatures$ function as below, in order to get the
        % feature description of a state:
        stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
        
        % act greedily according to the defined policy
        actionTaken = agent.predict(stateFeatures, 1);
              
        [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
            agentMovementHistory ] = ...
            actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
            currentTimeStep, agentMovementHistory, ...
            probabilityOfUniformlyRandomDirectionTaken ) ;
        
        Return = Return + agentRewardSignal;
       
        [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
            MDP, realAgentLocation, blockSize );
        
        currentMap = viewableGridMap ; %#ok<NASGU>
        
        refreshScreen
        
        pause(0.15)
        
    end
    
    currentMap = MDP ;
    agentLocation = realAgentLocation ;
    
    Return
    testReturns(e) = Return;
    
    printAgentTrajectory
    pause(1)
    
end % for each episode

disp("Final average return after " + numTestEpisodes + " episodes");
disp(mean(testReturns));

