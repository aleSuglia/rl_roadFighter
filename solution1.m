%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;
NUM_ACTIONS = 3;

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

tempGrid = [ roadBasisGridMaps(2).Grid; ...
  roadBasisGridMaps(3).Grid; roadBasisGridMaps(2).Grid; ...
  roadBasisGridMaps(8).Grid; roadBasisGridMaps(7).Grid ] ;

tempStart = [ n_MiniMapBlocksPerMap * blockSize, 1 ] ;

tempMarkerRescaleFactor = 1/( (25^2)/36 ) ;

MDP_1 = GridMap(tempGrid, tempStart, tempMarkerRescaleFactor, ...
    probabilityOfUniformlyRandomDirectionTaken) ;

% Appending a matrix (same size size as the grid) with the locations of 
% cars:
MDP_1.CarLocations = [0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     1     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     1     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     1     0     0     0 ; ...
                      0     0     1     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     1     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     1     0     0 ; ...
                      0     0     0     0     0 ];


% Appending the reward function (depends on next state and, only for 
% terminal states, on the current state):
MDP_1.RewardFunction = generateRewardFunction( MDP_1, rewards ) ;


%% Deterministic Policy to evaluate:
pi_test1 = UP * ones( MDP_1.GridSize ); % Default action: up.
pi_test1(:, 1) = UP_RIGHT; % When on the leftmost column, go up right.
pi_test1(:, 5) = UP_LEFT ; % When on the rightmost column, go up left.
pi_test1(:, 3) = UP_LEFT ; % When on the center column, go up left.

% Initialise the current agent with the specified policy and MDP
agent = Agent(MDP_1, pi_test1, MDP_1.Start, NUM_ACTIONS);

disp("Initial state values");
% the initial value function is initialised to 0 for each state
disp(agent.stateValues); 

[agent, numIterations] = agent.policyEvaluation(MDP_1, 1e-6);

disp("Final state values");
disp(agent.stateValues)
disp("Policy evaluation # iterations");
disp(numIterations)
    

