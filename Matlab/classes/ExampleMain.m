% ExampleScript.m

% Create a new figure handle
figureHandle = figure;

% Create a few shapes using the ShapeElement class
room1 = ShapeElement([0, 5, 5, 0], [0, 0, 5, 5], 'room');
obstacle1 = ShapeElement([2, 4, 4, 2], [2, 2, 3, 3], 'obstacle');
agent1 = ShapeElement([1, 1.5, 1, 0.5], [4, 4.5, 5, 4.5], 'agent');

% Discretize the shapes with a resolution of 0.2
resolution = 0.2;
room1.discretizeElement(resolution);
obstacle1.discretizeElement(resolution);
agent1.discretizeElement(resolution);

% Plot the shapes using the same figure handle
room1.plotShape(figureHandle);
obstacle1.plotShape(figureHandle);
agent1.plotShape(figureHandle);
