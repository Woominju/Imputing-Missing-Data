function dXdt = multiCoupledLorenz(t, X, sigma, rho, beta, epsilon, N)
    % multiCoupledLorenz: Compute the derivatives for a network of coupled Lorenz systems
    % Inputs:
    %   t       - Time variable (automatically passed by ode45, not used directly)
    %   X       - State vector of dimension 3*N x 1, ordered as [x1,y1,z1,x2,y2,z2,...,xN,yN,zN]
    %   sigma   - Parameter sigma of the Lorenz system
    %   rho     - Parameter rho of the Lorenz system
    %   beta    - Parameter beta of the Lorenz system
    %   epsilon - Coupling strength between subsystems
    %   N       - Number of coupled Lorenz subsystems
    % Output:
    %   dXdt    - Derivative of the state vector, same dimension as X

    % Initialize the derivative vector
    dXdt = zeros(3*N, 1);
    
    % Iterate over each Lorenz subsystem to compute derivatives
    for i = 1:N
        % Extract state variables for the i-th subsystem
        x_i = X(3*(i-1)+1);
        y_i = X(3*(i-1)+2);
        z_i = X(3*(i-1)+3);
        
        % Standard Lorenz equations (without coupling)
        dx = sigma * (y_i - x_i);
        dy = x_i * (rho - z_i) - y_i;
        dz = x_i * y_i - beta * z_i;
        
        % Add linear coupling term in the x-direction (diffusive coupling)
        if N > 1
            coupling_x = 0;
            for j = 1:N
                if j ~= i
                    x_j = X(3*(j-1)+1);
                    coupling_x = coupling_x + (x_j - x_i);
                end
            end
            dx = dx + epsilon * coupling_x;
        end
        
        % Assign computed derivatives to the output vector
        dXdt(3*(i-1)+1) = dx;
        dXdt(3*(i-1)+2) = dy;
        dXdt(3*(i-1)+3) = dz;
    end
end