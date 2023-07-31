% ShapeElement.m (Updated)
classdef ShapeElement < handle
    properties
        x_coords
        y_coords
        element_type
    end
    
    methods
        % Constructor to initialize the properties
        function obj = ShapeElement(x, y, type)
            obj.x_coords = x;
            obj.y_coords = y;
            obj.element_type = type;
        end
        
        % Updated method to discretize the element
        function discretizeElement(obj, resolution)
            x_ob = [];
            y_ob = [];
            
            % Add the first point to the end of the list to close the polygon
            x = [obj.x_coords obj.x_coords(1)];
            y = [obj.y_coords obj.y_coords(1)];

            % Loop through each side of the element
            for j = 1:length(x)-1
                % Calculate the number of steps needed to discretize the side
                length_side = sqrt((x(j) - x(j+1))^2 + (y(j) - y(j+1))^2);
                num_steps = ceil(length_side / resolution);

                % Add the discretized points to the vector
                x_ob = [x_ob linspace(x(j), x(j+1), num_steps)];
                y_ob = [y_ob linspace(y(j), y(j+1), num_steps)];
            end
            
            % Update the x_coords and y_coords properties with the discretized coordinates
            obj.x_coords = x_ob;
            obj.y_coords = y_ob;
        end
        
        % Method to plot the shape inside the class
        function plotShape(obj, figHandle)
            if nargin < 2
                figure;
                figHandle = gca;
            else
                figure(figHandle);
            end
            
            hold on;
            plot([obj.x_coords obj.x_coords(1)], [obj.y_coords obj.y_coords(1)], '-o', 'MarkerSize', 5);
            xlabel('X-coordinate');
            ylabel('Y-coordinate');
            title('Shape');
            grid on;
            hold off;
        end
    end
end
