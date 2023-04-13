%% Import the CSV and create a cell variable for the shape
clear
clc
addpath('Rooms/');
T = readtable('createGrid().csv', 'ReadVariableNames',false, 'Delimiter',',', 'HeaderLines',1, 'TreatAsEmpty',{'NA','na'});
shape = cell(height(T),2);
x_max = -1e5;
y_max = -1e5;
x_min = 1e5;
y_min = 1e5;

%build the shape into a cell variable
for i = 1:height(T)
    disp(string(T{i,2}))
    x = str2num(string(T{i,2}));
    y = str2num(string(T{i,3}))*-1;
    % plot([x, x(1)],[y, y(1)])
    shape{i,1} = x;
    shape{i,2} = y;

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

%% normalize coordinates from 0 to 1 in both axes (square)
width_room = abs(x_max-x_min);
height_room = abs(y_max-y_min);
aspect_ratio = width_room/height_room;

x_scale = [0 width_room];
y_scale = [0 height_room];


for j = 1:width(shape)
    for i = 1:height(shape)
        if j == width(shape)
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
%     h=text(mean(shape{i,1}), mean(shape{i,2}), {num2str(i)});
%     set(h,'color','r')
end
daspect([1 1 1])
grid on
grid minor
hold off