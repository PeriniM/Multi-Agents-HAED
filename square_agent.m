clear all;
% Define the dimensions of the room
room_width = 20;   % meters
room_height = 20;  % meters

% Define the positions of the three anchors
anchor1_position = [-room_width/2, -room_height/2];
anchor2_position = [room_width/2, -room_height/2];
anchor3_position = [-room_width/2, room_height/2];

% Define the position of the UWB sensor
sensor_position = [-5, -5];

% Create a figure window to visualize the room and the robot
figure('Position', [100 100 800 800]);
xlim([-room_width/2 room_width/2]);
ylim([-room_height/2 room_height/2]);

% Define initial position and orientation of the robot
x = 0;   % meters
y = 0;   % meters
theta = rand * 2*pi;   % radians

% Define the maximum speed and angular velocity of the robot
max_speed = 0.5;   % meters per second
max_angular_velocity = pi/2;   % radians per second

% Define the time step for the simulation
dt = 0.01;   % seconds
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
    x = x + vx * dt;
    y = y + vy * dt;
    theta = theta + omega * dt;
    
    % Update the graphics object for the robot
    square_width = 1;   % meters
    square_vertices = [square_width/2 square_width/2; -square_width/2 square_width/2; -square_width/2 -square_width/2; square_width/2 -square_width/2];
    rotation_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    rotated_vertices = (rotation_matrix * square_vertices')';
    rotated_vertices(:,1) = rotated_vertices(:,1) + x;
    rotated_vertices(:,2) = rotated_vertices(:,2) + y;
    if exist('h_square', 'var')
        set(h_square, 'XData', rotated_vertices(:,1), 'YData', rotated_vertices(:,2));
    else
        h_square = fill(rotated_vertices(:,1), rotated_vertices(:,2), 'b');
    end
    
    % Calculate the distances between the robot and the anchors
    d1 = norm(sensor_position - anchor1_position);
    d2 = norm(sensor_position - anchor2_position);
    d3 = norm(sensor_position - anchor3_position);
    
    % Perform triangulation to find the position of the robot
    p1 = anchor1_position';
    p2 = anchor2_position';
    p3 = anchor3_position';
    A = 2*[p2(1)-p1(1) p2(2)-p1(2); p3(1)-p1(1) p3(2)-p1(2)];
    b = [d1^2-d2^2-p1(1)^2-p1(2)^2+p2(1)^2+p2(2)^2; d1^2-d3^2-p1(1)^2-p1(2)^2+p3(1)^2+p3(2)^2];
    x_uwb = (A\b)';
    
    % Update the position of the UWB sensor
    sensor_position = x_uwb;
    
    % Update the graphics object for the UWB sensor
    if exist('h_sensor', 'var')
        set(h_sensor, 'XData', sensor_position(1), 'YData', sensor_position(2));
    else
        h_sensor = plot(sensor_position(1), sensor_position(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    end
    
    % Update the graphics object for the anchors
    if exist('h_anchor1', 'var')
        set(h_anchor1, 'XData', anchor1_position(1), 'YData', anchor1_position(2));
    else
        h_anchor1 = plot(anchor1_position(1), anchor1_position(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
    end
    
    if exist('h_anchor2', 'var')
        set(h_anchor2, 'XData', anchor2_position(1), 'YData', anchor2_position(2));
    else
        h_anchor2 = plot(anchor2_position(1), anchor2_position(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
    end
    
    if exist('h_anchor3', 'var')
        set(h_anchor3, 'XData', anchor3_position(1), 'YData', anchor3_position(2));
    else
        h_anchor3 = plot(anchor3_position(1), anchor3_position(2), 'gx', 'MarkerSize', 10, 'LineWidth', 2);
    end
    xlim([-room_width/2 room_width/2]);
    ylim([-room_height/2 room_height/2]);
    % Pause for a short time to simulate real-time behavior
    pause(dt);
end
hold off