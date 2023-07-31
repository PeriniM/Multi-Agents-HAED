function [x_ob, y_ob] = discretizeElement(x, y, resolution)
    % This function discretizes a 2D element into a series of points equally on both axes
    
    % Add the first point to the end of the list to close the polygon
    x = [x x(1)];
    y = [y y(1)];
    x_ob = [];
    y_ob = [];

    % Loop through each side of the element
    for j = 1:length(x)-1

        % Calculate the number of steps needed to discretize the side
        length_side = sqrt((x(j)-x(j+1))^2 + (y(j)-y(j+1))^2);
        num_steps = ceil(length_side/resolution);

        % Add the discretized points to the vector
        x_ob = [x_ob linspace(x(j), x(j+1), num_steps)];
        y_ob = [y_ob linspace(y(j), y(j+1), num_steps)];
    end
end