import numpy as np
from typing import Callable, Tuple
# from lower_solve import lower_problem_solve
from sys_model import DoubleIntegrator
from config import Config
from numpy.linalg import pinv
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

import json
import os
import pickle

class BilevelOptimizer:
    def __init__(
        self,
        u_star: Callable,
        z_ini: Callable,
        upper_lr: float = 1e-10,
        lower_lr: float = 1e-9,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        lower_iterations: int = 50
    ):
        """
        Initialize bi-level optimizer.
        """
        #self.upper_objective = upper_objective
        self.u_star = u_star
        self.z_ini = z_ini

        self.upper_lr = upper_lr
        self.lower_lr = lower_lr
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.lower_iterations = lower_iterations

    def vech_to_matrix(self, x):
        """
        Convert a vector x = vech(A) back to the symmetric matrix A.
        """
        n = int((-1 + np.sqrt(1 + 8*len(x)))/2)
        
        # Initialize the output matrix
        A = np.zeros((n, n))
        
        # Fill in the lower triangular part
        idx = 0
        for i in range(n):
            for j in range(i, n):
                A[i,j] = x[idx]
                idx += 1
        
        # Make the matrix symmetric by copying lower triangle to upper triangle
        A = A + A.T - np.diag(np.diag(A))
        
        return A

    def vech(self, A):
        """
        Calculate the half-vectorization (vech) of a symmetric matrix A.
        """
        # Check if matrix is symmetric
        if not np.allclose(A, A.T):
            raise ValueError("Input matrix must be symmetric")
        
        n = A.shape[0]
        # Get indices for lower triangular part (including diagonal)
        indices = np.triu_indices(n)
        return A[indices]
    
    def project_to_psd(self, A, tol=1e-10):
        """
        Project matrix onto the positive semidefinite cone.
        """
        # Set negative eigenvalues to 0
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Reconstruct the matrix
        A_proj = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return A_proj

    def optimize_lower_level(self, theta, y_init, K, lower_lr_set):
        """
        Optimize lower-level problem for fixed upper-level variable u(theta).
        Solve the forward optimal control
        """
        config = Config()
        p = config.p
        m = config.m
        T_ini = config.T_ini

        vech_q = theta[:int(p*(p+1)/2)]
        vech_r = theta[int(p*(p+1)/2):]
        Q = self.vech_to_matrix(vech_q)
        R = self.vech_to_matrix(vech_r)
        q_block = [Q] * config.N
        q_cal = block_diag(*q_block)
        r_block = [R] * config.N
        r_cal = block_diag(*r_block)

        Kp = K[:, :(m+p)*T_ini]
        Kf = K[:, (m+p)*T_ini:]

        def compute_cost(u):
            z = np.concatenate([self.z_ini, u])
            KT_Q_K = K.T @ q_cal @ K
            return z.T @ KT_Q_K @ z + u.T @ r_cal @ u
            
        def compute_gradient(u):
            return (Kf.T @ q_cal @ Kf + r_cal) @ u + Kf.T @ q_cal.T @ Kp @ self.z_ini

        Hess = Kf.T @ q_cal @ Kf + r_cal
        # print(np.all(np.linalg.eigvalsh(Hess) >= -1e-10))

        cost_history = []
        max_iterations=10000
        # Gradient descent
        uo = y_init
        # for i in range(max_iterations):
        #     grad = compute_gradient(uo)
        #     u_new = uo - learning_rate * grad

        #     # Compute cost
        #     current_cost = compute_cost(u_new)
        #     cost_history.append(current_cost)
                
        #     # Check convergence
        #     if np.linalg.norm(grad) < 1e-6:
        #         print("lower converge stop")
        #         break
                
        #     uo = u_new
        grad = compute_gradient(uo)
        grad_norm = np.linalg.norm(grad)
        grad_history = []
        iteration = 0
        flag = 0
        lower_lr = lower_lr_set
        while grad_norm > 1e-5:
            grad = compute_gradient(uo)
            grad_norm = np.linalg.norm(grad)
            grad_history.append(grad_norm)
            u_new = uo - lower_lr * grad

            # Compute cost 
            current_cost = compute_cost(u_new)
            cost_history.append(current_cost)
            iteration += 1
            if(iteration > 3e5):
                print(f"max iter at {iteration}!! the grad is {grad_norm}")
                # if(grad_norm > 1e-4):
                if(flag < 2):
                    lower_lr = lower_lr * 2
                    flag = flag + 1
                    print("lower_lr up")
                break
            uo = u_new.copy()

        # plt.plot(np.squeeze(grad_history), '.-')
        # plt.show()
        return uo, cost_history, lower_lr
    
    def _make_hankel(self, x, L):
        m, K = x.shape
        H = np.zeros((m * L, K - L + 1))
        for i in range(K - L + 1):
            H[:, i] = x[:, i:i + L].reshape(-1, order='F')
        return H

    def process_K(self):
        system = DoubleIntegrator()
        config = Config()
        T_ini = config.T_ini
        N = config.N
        m = config.m
        p = config.p
        x_data = system.xs
        u_data = np.array([])
        y_data = np.array([])

        max_traj_len = 50
        for i in range(max_traj_len):
            u = (np.random.rand(system.nu, 1) * 2) * 0.1
            u_data = np.hstack([u_data, u]) if u_data.size else u
            y = system.model['h'](x_data.reshape(-1,1), u)
            y_data = np.hstack([y_data, y]) if y_data.size else y
            x_data = system.model['f'](x_data.reshape(-1,1), u)

        U = self._make_hankel(u_data, T_ini+N)
        Y = self._make_hankel(y_data, T_ini+N)
        Up = U[:T_ini*m, :]
        Uf = U[T_ini*m:, :]
        Yp = Y[:T_ini*p, :]
        Yf = Y[T_ini*p:, :]
        K = Yf @ np.linalg.pinv(np.vstack([Up, Yp, Uf]))
        return K


    def process_phi(self, uo, K, omega_ini):  # calculate matrix Phi
        """
        Generate matrix Phi with collected data
        """
        config = Config()
        T_ini = config.T_ini
        N = config.N
        m = config.m
        p = config.p
        Kp = K[:, :(m+p)*T_ini]
        Kf = K[:, (m+p)*T_ini:]
        
        I_kf = np.eye(Kf.T.shape[0])
        omega_ini = np.reshape(omega_ini, (-1, 1), order='F')
        
        y_hat = Kf @ uo + Kp @ omega_ini
        
        # Calculate phi matrices
        phi_y = np.zeros((Kf.T.shape[0], p*p))
        phi_u = np.zeros((I_kf.shape[0], m*m))

        for i in range(N):
            y_slice = y_hat[i*p:(i+1)*p]
            K_slice = Kf[i*p:(i+1)*p, :].T
            phi_y += np.kron(y_slice.T, K_slice)
        
        for i in range(N):
            u_slice = uo[i*m:(i+1)*m]
            I_slice = I_kf[i*m:(i+1)*m, :].T
            phi_u += np.kron(u_slice.T, I_slice)
        
        phi = np.hstack([phi_y, phi_u])
        # phi_rank = np.linalg.matrix_rank(phi, tol=1e-6)
        
        # Calculate phi_v
        phi_v = np.column_stack([
            phi_y[:, 0],
            phi_y[:, 1] + phi_y[:, 3],
            phi_y[:, 2] + phi_y[:, 6],
            phi_y[:, 4],
            phi_y[:, 5] + phi_y[:, 7],
            phi_y[:, 8],
            phi_u[:, 0],
            phi_u[:, 1] + phi_u[:, 2],
            phi_u[:, 3]
        ])
        
        # Calculate rank of phi_v
        phi_v_rank = np.linalg.matrix_rank(phi_v, tol=1e-4)
        #print("rank:", phi_v_rank)
        
        return phi_v, Kf
    
    def cal_q(self, Kf, x_curr, phi):
        config = Config()
        p = config.p
        vech_q = x_curr[:int(p*(p+1)/2)]
        vech_r = x_curr[int(p*(p+1)/2):]
        Q = self.vech_to_matrix(vech_q)
        R = self.vech_to_matrix(vech_r)
        q_block = [Q] * config.N
        q_cal = block_diag(*q_block)
        r_block = [R] * config.N
        r_cal = block_diag(*r_block)
        
        # d_theta_u = np.zeros([config.m*config.N, x_curr.shape[0]]) 
        max_iter = 10000
        pinv_lr = 0.000001
        A = Kf.T @ q_cal @ Kf + r_cal
        B = phi
        # pin_cal = pinv(A)@B
        # print(pin_cal)
        # print(np.linalg.solve(A, B))
        # res_his = []
        # for i in range(max_iter):  ##################### not convergence! ################### 

        #     gradient = 2 * A.T @ (A @ d_theta_u - B)
        #     d_theta_u = d_theta_u - pinv_lr * gradient
            
        #     # Compute the norm of the residual
        #     residual_norm = np.linalg.norm(A @ d_theta_u - B)
        #     res_his.append(residual_norm)
        #     # Check if the residual is within the tolerance
        #     if residual_norm <= 1e-5:
        #         print(f"Converged after {i+1} iterations.")
        #         return d_theta_u
            
        # # If max_iter is reached without convergence
        # print(f"Did not converge within {max_iter} iterations.")
        # print(d_theta_u)
        # plt.plot(np.squeeze(res_his), '.-')
        # plt.show
        return np.linalg.solve(A, B)

    
    def optimize(
        self, 
        x_init: np.ndarray, 
        y_init: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Perform bi-level optimization.
        """
        x_current = x_init.copy()
        y_current = y_init.copy()
        upper_loss_history = []
        upper_grad_history = []
        batch_loss = 1e5

        config = Config()
        p = config.p
        K = self.process_K()
        llr = self.lower_lr
        for i in range(self.max_iterations):
            # print(i)
            # Solve lower-level problem
            y_optimal, lower_cost, llr = self.optimize_lower_level(x_current, y_current, K, llr)
            # plt.plot(np.squeeze(lower_cost), '.-')
            # plt.show()
            phi, Kf = self.process_phi(y_optimal, K, self.z_ini)

            # Calculate upper-level gradient
            upper_grad = (self.u_star - y_optimal).reshape(1, -1) @ self.cal_q(Kf, x_current, phi)
            # a = self.u_star - y_optimal
            # b = self.cal_guu_pinv(Kf, x_current)@ phi
            # print(self.upper_lr * upper_grad.T)
            
            # Update upper-level variables
            x_new = x_current - self.upper_lr * upper_grad.T

            # projection on PSD
            vech_q_new = x_new[:int(p*(p+1)/2)]
            vech_r_new = x_new[int(p*(p+1)/2):]
            Q_new = self.vech_to_matrix(vech_q_new)
            R_new = self.vech_to_matrix(vech_r_new)
            proj_Q = self.project_to_psd(Q_new)
            proj_R = self.project_to_psd(R_new)
            x_new_proj = np.hstack((self.vech(proj_Q), self.vech(proj_R))).T
            # print(np.all(np.linalg.eigvalsh(proj_Q) >= -1e-10))
            # x_new_proj =x_new
            
            # Record losses
            # upper_loss = self.upper_objective(x_current, y_optimal)
            upper_loss = np.linalg.norm(self.u_star - y_optimal, 2) / np.linalg.norm(self.u_star, 2)  # here is the last step loss
            upper_loss_history.append(upper_loss)
            upper_grad_history.append(np.linalg.norm(upper_grad))
            # Check convergence
            if np.all(np.abs(upper_grad) < self.tolerance):
                print(f"Converged after {i+1} iterations!")
                break
                
            # Update current values
            x_current = x_new_proj.reshape(-1, 1)
            y_current = y_optimal.copy()  # update or not both are ok
            
            # Print progress
            if (i + 1) % 100 == 0:
                llr = self.lower_lr
                print(f"Iteration {i+1}:")
                print(f"Upper-level loss: {upper_loss:.6f}")
                if(upper_loss < 0.025):
                    self.upper_lr = self.upper_lr
                    print("optimization is done")
                else:
                    if(upper_loss - batch_loss > 0):
                        self.upper_lr = self.upper_lr*0.5
                        print("turn down")
                    if(0 < batch_loss - upper_loss < 5e-2):
                        self.upper_lr = self.upper_lr*2
                        print("turn up")
                batch_loss = upper_loss
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(upper_grad_history, '.-')
        plt.savefig("./grad.jpg")

        # 存储upper_grad
        file_path = "./upper_grad.json"
        if not os.path.exists(file_path):
            all_data = []
        else:
            with open(file_path, 'r') as f:
                all_data = json.load(f)
        all_data.append(upper_grad_history)
        with open(file_path, 'w') as f:
            json.dump(all_data, f)

        return x_current, y_optimal, upper_loss_history



if __name__ == "__main__":
    
    opt_traj = np.load('./opt_traj_rad_ini3.npy', allow_pickle=True).item()
    _u_star = np.reshape(opt_traj['u_opt'],(-1, 1), order='F')
    # if it has noises
    # _u_star = _u_star + np.random.normal(0, 0.01, size=_u_star.shape)

    # Initialize optimizer
    optimizer = BilevelOptimizer(
       # upper_objective=upper_objective,
        u_star = _u_star,
        z_ini = np.reshape(opt_traj['z_ini'],(-1, 1), order='F'),
        upper_lr=1,
        lower_lr=1e-5
    )
    
    # Initial guesses
    config = Config()
    var_num = (config.p*(config.p+1)+config.m*(config.m+1))/2
    x_init = np.random.rand(int(var_num), 1)  # Q,R guess
    print("Q,R guess:")
    print(x_init)
    y_init = np.zeros([config.m*config.N,1])  # uo guess
    
    # Optimize
    x_opt, y_opt, upper_history = optimizer.optimize(x_init, y_init)
    
    print("\nOptimization Results:")
    # print(f"QR_estimate: {x_opt}")
    #print(f"u_final: {y_opt}")
    #print(f"u_expert: {opt_traj['u_opt']}")
    print(f"Q: {optimizer.vech_to_matrix(x_opt[:int(3*(3+1)/2)])}")
    print(f"R: {optimizer.vech_to_matrix(x_opt[int(3*(3+1)/2):])}")

    plt.xscale("log")
    plt.yscale("log")
    plt.plot(upper_history, '.-')
    plt.savefig("./cost.jpg")
    # 存储upper_history
    file_path = "./upper_history.json"
    if not os.path.exists(file_path):
        all_data = []
    else:
        with open(file_path, 'r') as f:
            all_data = json.load(f)
    all_data.append(upper_history)
    with open(file_path, 'w') as f:
        json.dump(all_data, f)
            
    #print(f"Final upper-level objective: {upper_objective(x_opt, y_opt):.6f}")
    plt.figure()
    uo_plot = y_opt.reshape(2,-1,order='F')
    plt.plot(uo_plot[0, :], uo_plot[1, :], '.-')
    plt.savefig("./uo.jpg")
    # 存储generated input: uo
    file_path = "./uo_log.json"
    if not os.path.exists(file_path):
        all_data = []
    else:
        with open(file_path, 'r') as f:
            all_data = json.load(f)
    all_data.append(uo_plot.tolist())
    with open(file_path, 'w') as f:
        json.dump(all_data, f)

