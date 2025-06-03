function sys = double_integrator()
    sys.name = "double-integrator";
    %% State-space Matrices
    sys.nx = 3;
    sys.nu = 2;
    sys.ny = 3;
    
    % sys.model.A = [0.8147 ,   0.2134;
    % 0.2058 ,   0.6324];
    % sys.model.B = [0.0975; 0.2785];
    % sys.model.C = [1, 0; 0, 1];
    % sys.model.D = 0;

    sys.model.A = [0.8147  ,  0.1270  ,  0.3324;
    0.2058  ,  0.9134  ,  0.0975;
    0.1270  ,  0.2324  ,  0.6785];
    % sys.model.A = eye(sys.nx);
    sys.model.B = [0.5469  ,  0.9575;
    0.9157  ,  0.9649;
    0.7577  ,  0.1576];
    sys.model.C = [0.8003  ,  0.4218  ,  0.9157;
    0.1419  ,  0.9157  ,  0.7922;
    0.6557  ,  0.7922  ,  0.9595];
    sys.model.D = [0.0357  ,  0.8491;
    0.8491  ,  0.9340;
    0.9340  ,  0.6787];
    % sys.model.C = eye(sys.nx);
    % sys.model.D = zeros(3,2);

    sys.model.f = @(x, u) sys.model.A*x + sys.model.B*u;  % dynamics
    sys.model.h = @(x, u) sys.model.C*x + sys.model.D*u;  % measurement
    
    %% Constraints
    sys.u_max = 1;
    sys.y_max = 4;
    % sys.constraints.Hx = [eye(sys.nx); -eye(sys.nx)];
    % sys.constraints.hx = ones(2*sys.nx, 1) * 4;
    sys.constraints.Hu = [eye(sys.nu); -eye(sys.nu)];
    sys.constraints.hu = ones(2*sys.nu, 1) * sys.u_max;
    sys.constraints.Hy = [eye(sys.ny); -eye(sys.ny)];
    sys.constraints.hy = ones(2*sys.ny, 1) * sys.y_max;
    
    %% Initial Condition
    % If the second state is chosen not to be 0, be careful constructing 
    % the initial trajectory
    sys.xs = [-2.5; 1; 0.5];
    sys.us = [0; 0];
    sys.ys = sys.model.h(sys.xs, sys.us);
    
    %% Equilibrium
    sys.xf = [0; 0; 0];
    sys.uf = [0; 0];
    sys.yf = [0; 0; 0];
end