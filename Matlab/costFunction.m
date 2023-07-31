function f0 = costFunction(initial_guess, anchors, distances, weights)

    % perform multilateration to find the location of the tag

    % initial_guess is a 1x2 vector containing the initial guess for the tag location
    % anchor_coords is a 4x2 matrix containing the coordinates of the anchors
    % distances is a 1x4 vector containing the distances from the tag to the anchors
    % weights is a 1x6 vector containing the weights for the equations

    % f0 is the value of the objective function at the solution

    eq = [];
    for i = 1:length(distances)
        eq = [eq, (initial_guess(1)-anchors{i}(1))^2 + (initial_guess(2)-anchors{i}(2))^2 - distances(i)^2];
    end
    
    eq = weights.*eq.^2;
    f0 = sum(eq);

    end