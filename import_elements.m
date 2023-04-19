%% Import the CSV and create a cell variable for the shape
clear
clc
addpath('Rooms/');
T = readtable('createGrid().csv', 'ReadVariableNames',false, 'Delimiter',',', 'HeaderLines',1, 'TreatAsEmpty',{'NA','na'});
shape = cell(height(T),3);
x_max = -1e5;
y_max = -1e5;
x_min = 1e5;
y_min = 1e5;

%build the shape into a cell variable
for i = 1:height(T)
    % disp(string(T{i,2}))
    x = str2num(string(T{i,2}));
    y = str2num(string(T{i,3}))*-1;
    element_type = string(T{i,5});
    % plot([x, x(1)],[y, y(1)])
    shape{i,1} = x;
    shape{i,2} = y;
    shape{i,3} = element_type;

    % takes min and max of both axes for normalization
    if x_max<max(x)
        x_max = max(x);
    end
    if y_max<max(y)
        y_max = max(y);
    end
    if x_min>min(x)
        x_min = min(x);
    end
    if y_min>min(y)
        y_min = min(y);
    end
end

%% normalize coordinates in both axes
width_room = abs(x_max-x_min);
height_room = abs(y_max-y_min);
aspect_ratio = width_room/height_room;

x_scale = [0 width_room];
y_scale = [0 height_room];

for j = 1:width(shape)-1
    for i = 1:height(shape)
        if j == width(shape)-1
            % [A, B] --> [a, b] --> (val - A)*(b-a)/(B-A) + a
            shape{i,j} = (shape{i,j} - y_min)*(y_scale(2)-y_scale(1))/(y_max-y_min) + y_scale(1);
        else
            shape{i,j} = (shape{i,j} - x_min)*(x_scale(2)-x_scale(1))/(x_max-x_min) + x_scale(1);
        end
    end
end

%% plot normalized shape
figure(1)
hold on
for i = 1:height(shape)
    %reconnect to initial point
    plot([shape{i,1}(1,:) shape{i,1}(1,1)], [shape{i,2}(1,:) shape{i,2}(1,1)])
end
daspect([1 1 1])
grid on
grid minor
hold off

%% add some uwb anchors at the corners of the room
% add anchors at the vertices coordinates of the room

agent_vert = {};
obstacle_vert = {};
room_vert = {};

for i = 1:height(shape)
    if shape{i,3} == "room"
        % append the room coordinates to the room_vert cell
        room_vert{end+1, 1} = shape{i,1};
        room_vert{end, 2} = shape{i,2};

    elseif shape{i,3} == "obstacle"
        % append the obstacle coordinates to the obstacle_vert cell
        obstacle_vert{end+1, 1} = shape{i,1};
        obstacle_vert{end, 2} = shape{i,2};

    elseif shape{i,3} == "agent"
        % append the agent coordinates to the agent_vert
        agent_vert{end+1, 1} = shape{i,1};
        agent_vert{end, 2} = shape{i,2};
        agent_vert{end, 3} = [mean(agent_vert{end,1}) mean(agent_vert{end,2})];
    end
end

%% discretize elements

resolution = 0.5;

% discretize the obstacles
for i = 1:height(obstacle_vert)
    % discretize the obstacle and room
    [obstacle_vert{i,3}, obstacle_vert{i,4}] = discretizeElement(obstacle_vert{i,1}, obstacle_vert{i,2}, resolution);
end

% discretize the room
for i = 1:height(room_vert)
    % discretize the obstacle and room
    [room_vert{i,3}, room_vert{i,4}] = discretizeElement(room_vert{i,1}, room_vert{i,2}, resolution);
end

%% plot discretized shape
figure(2)
hold on

% plot the obstacles
for i = 1:height(obstacle_vert)
    %reconnect to initial point
    plot([obstacle_vert{i,3}(1,:) obstacle_vert{i,3}(1,1)], [obstacle_vert{i,4}(1,:) obstacle_vert{i,4}(1,1)], 'o', 'MarkerSize', 2)
end

% plot the room
for i = 1:height(room_vert)
    %reconnect to initial point
    plot([room_vert{i,3}(1,:) room_vert{i,3}(1,1)], [room_vert{i,4}(1,:) room_vert{i,4}(1,1)], 'o', 'MarkerSize', 2)
end

daspect([1 1 1])
grid on
grid minor
hold off

%% add the uwb anchors at the corners of the room

anchors = {};

% add anchors at the vertices coordinates of the room
for i = 1:height(room_vert)
    % for each vertex of the room, add an anchor
    for j = 1:length(room_vert{i,1})
        % add the anchor to the anchors cell
        anchors{end+1, 1} = [room_vert{i,1}(j) room_vert{i,2}(j)];
    end
end

% plot the anchors
figure(2)
hold on
for i = 1:height(anchors)
    plot(anchors{i}(1), anchors{i}(2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r')
end

hold off

%% add unicycle dynamics to the agents

% Define initial position and orientation of the agent

x_CoM = agent_vert{1,3}(1);
y_CoM = agent_vert{1,3}(2);
theta = rand * 2*pi;   % radians

% Define the maximum speed and angular velocity of the agent
max_speed = 2;   % meters per second
max_angular_velocity = pi;   % radians per second


% Define the time step for the simulation
dt = 0.01;   % seconds

agent_vertices = [];

% translate the center of mass to the origin
for i = 1:length(agent_vert{1,1})
    agent_vertices(i,1) = agent_vert{1,1}(i) - x_CoM;
    agent_vertices(i,2) = agent_vert{1,2}(i) - y_CoM;
end

% guesses for center of mass uwb estimation
CoM_guess = [0 0];
% zeros vector for the weights of the minimization
estimate_weights = ones(1, length(anchors)).*10;

hold on
% Loop through the simulation
while true
    % Generate random velocity and angular velocity for the robot
    speed = rand * max_speed;
    angular_velocity = (rand - 0.5) * 2 * max_angular_velocity;
    
    % Calculate the x and y components of the velocity and orientation
    vx = speed * cos(theta);
    vy = speed * sin(theta);
    omega = angular_velocity;
    
    % Update the position and orientation of the robot
    x_CoM = x_CoM + vx * dt;
    y_CoM = y_CoM + vy * dt;
    theta = theta + omega * dt;

    % Update the graphics object for the robot
    
    rotation_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    rotated_vertices = (rotation_matrix * agent_vertices')';
    rotated_vertices(:,1) = rotated_vertices(:,1) + x_CoM;
    rotated_vertices(:,2) = rotated_vertices(:,2) + y_CoM;

    if exist('h_agent', 'var')
        set(h_agent, 'XData', rotated_vertices(:,1), 'YData', rotated_vertices(:,2));
    else
        h_agent = fill(rotated_vertices(:,1), rotated_vertices(:,2), 'b');
    end
    
    % Update the graphics object for the robot's center of mass
    if exist('h_CoM', 'var')
        set(h_CoM, 'XData', x_CoM, 'YData', y_CoM);
    else
        h_CoM = plot(x_CoM, y_CoM, 'o', 'MarkerSize', 3, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
    end

    % Perform trilateration to estimate the position of the robot's center of mass
    
    % for each anchor
    for i = 1:height(anchors)
        % calculate the distance between the robot and the anchor
        distance(i) = norm([x_CoM y_CoM] - anchors{i});
    end

    % minimize the objective function using fminsearch
    options = optimset('TolX',1e-5,'MaxFunEvals',10000,'MaxIter',10000);
    CoM_estim = fminsearch(@(x) costFunction(x, anchors, distance, estimate_weights), CoM_guess, options);
    
    % update the guess for the next iteration
    CoM_guess = CoM_estim;

    % update the graphics connecting the robot's estimated position to all the anchors
    if exist('h_CoM_estim', 'var')
        for i = 1:height(anchors)
            set(h_CoM_estim(i), 'XData', [CoM_estim(1) anchors{i}(1)], 'YData', [CoM_estim(2) anchors{i}(2)]);
        end
    else
        h_CoM_estim = [];
        for i = 1:height(anchors)
            h_CoM_estim(i) = plot([CoM_estim(1) anchors{i}(1)], [CoM_estim(2) anchors{i}(2)], 'r');
        end
    end

    % Pause for a short time to simulate real-time behavior
    pause(dt);
end
hold off
