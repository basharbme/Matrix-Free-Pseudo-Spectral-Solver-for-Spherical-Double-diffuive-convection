#!/usr/bin/env python

import numpy as np

from scipy.fftpack import dct, idct, dst, idst

from scipy.sparse import bmat, diags, block_diag, eye, triu, identity
from scipy.sparse.linalg import expm, inv, spsolve
from scipy.linalg import solve_triangular
import scipy

import sys, os, time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla


#----------- Latex font ----
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#------------------------------
import matplotlib.pyplot as plt

from B_Matrix import *

def cheb_radial(N,d):
	r_i = 1.0; r_o = 1.0+d;

	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		x = 0.5*(r_o + r_i) + 0.5*(r_o-r_i)*x; # Transform to radial

		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	
	return D, x.reshape(N+1);


#~~~~~~~~~ Nr x N_theta ~~~~~~~~~~~~~
# Test Parameters to pass
l = 2; d = 3.0;
Ra_c = 380.0; Ra_s = 50.0; S = Ra_s/Ra_c;
Ra = Ra_c*(1 + 0.25)
Tau = 5.0; Pr = 1.0;

Lx = np.pi;#2.0*(4.0*np.pi) # = length of spatial domain along x-direction
Frames = 10;
N_steps = 1000;
nsave = N_steps/Frames;
print "nsave ",nsave
N_fm = 2*25; # EVEN = 2*no. of fourier modes, 
Nr = 15; #no. of Chebyshev modes
D,R = cheb_radial(Nr,d); nr = len(R[1:-1]);
N = nr*N_fm
dt_newton = 1e02;  #default time step
dt = 1e-02;  #default time step

#X =  0.001*np.random.randn(3*nr*N_fm,1)
Stringy = ['Test_l2_10P']; # New-file
'''

Tau = 1.0/15.0; Pr = 1.0; 
l = 20.0;  d = 0.16; 
Ra_c = 541137.499805; Ra_s = 10**4; S = Ra_s/Ra_c;

#eps_r = 0.175530868966; # Location of real subcritical bifurcation
#eps_i = 0.129; # Approximately

eps = 0.1; # Maybe crit
Ra = Ra_c*(1.+eps); 

# Grid and required matrices:
Lx = np.pi;#2.0*(4.0*np.pi) # = length of spatial domain along x-direction
Frames = 100;
N_steps = (10**5); #10*(10**5); # 0.06 s/iter
nsave = N_steps/Frames;
print "nsave ",nsave
N_fm = 2*125; # EVEN = 2*no. of fourier modes, 
Nr = 15; #no. of Chebyshev modes
D,R = cheb_radial(Nr,d); nr = len(R[1:-1]);
N = nr*N_fm;
dt = 5e-05;  #default time step
dt_newton = 50.0; 
'''
# 20 times units
#print "Total Time = ",N_steps*(60.0/(10**3))*(1.0/3600.0)," hrs \n"
#sys.exit()

Stringy = ['Test2_l21_1P']; # New-file

#X =  0.001*np.random.randn(3*nr*N_fm,1)
X1 =  np.load("/home/mannixp/Dropbox/Leeds/Test2_l21_1P/X_VID_l20.npy")[-1];
#X =  np.load("/home/ma/p/pm4615/SPECTRAL/Test_l20_10P/X_VID_l20.npy")[-1];


# Use correct grid for DCT-II
x = np.zeros(N_fm);
for n in xrange(N_fm):
	x[n] = Lx*( (2*n+1.0)/(2.0*N_fm) ); # x in [0,L] 

# 1) Spatial wavenumber, k*pi/L# if L = pi just k
kxC = np.arange(0,N_fm); alphaC = kxC*np.pi/Lx;
kxS = np.arange(1,N_fm+1); alphaS = kxS*np.pi/Lx; 

# 2) Normalization Weights, must be inverses
akc = np.zeros(N_fm); aks = np.zeros(N_fm);
for k in range(N_fm):
	ks = k + 1;
	kc = k; 
	akc[k] = 1.0/dct(np.cos(kc*x),type=2,norm='ortho')[k]; # cosine
	aks[k] = 1.0/dst(np.sin(ks*x),type=2,norm='ortho')[k]; # sin


# 1) ~~~~~~~~~ Build Matrices ~~~~~~~~~

print "Building Matrices .... \n"

# a) Define matrices for Euler-Implicit #A = M - dt*L;
#~~~~~~~~~#~~~~~~~~~~~#~~~~~~~~~~~~#~~~
Rsq = R2(R,N_fm); #I = np.eye(nr*N_fm); 
A2 = NABLA2_SINE(D,R,N_fm);
A4 = Pr*NABLA4(D,R,N_fm);
NABLA2 = NABLA2_COS(D,R,N_fm); # Multplied by r^2

# b) Calculate matrices for Non-linear
#~~~~~~~~~#~~~~~~~~~~~#~~~~~~~~~~~~#~~~	
JT = J_theta(nr,N_fm); 
A2_SINE = NABLA2_SINE_NL(D,R,N_fm); 
gr = Pr*kGR(R,N_fm); # Sparse 
T0_JT = T0J_theta(R,N_fm,d); # Sparse 

def Matrices(dt):

	L_psi = A2.todense() - dt*A4;
	L_T = Rsq - dt*NABLA2;
	L_C = Rsq - Tau*dt*NABLA2;
	
	if dt > 1.0:
		print "GMRES iter mutlip by dt* .... \n"

		# P = dt*(M - dt*L)^{-1} Should be ... dt*(I - dt*L)^{-1}
		LINV_psi = dt*np.linalg.inv(L_psi);
		LINV_T = dt*np.linalg.inv(L_T.todense());
		LINV_C = dt*np.linalg.inv(L_C.todense());

	else:	
		print "Time-iter .... \n"
		# (M - dt*L)^{-1}
		LINV_psi = np.linalg.inv(L_psi);
		LINV_T = np.linalg.inv(L_T.todense());
		LINV_C = np.linalg.inv(L_C.todense());

	print "Inverted Preconditioners .... \n"

	LINV_psi_SP = scipy.sparse.triu(LINV_psi, k=-nr, format="csr")
	LINV_T_SP = scipy.sparse.triu(LINV_T, k=-nr, format="csr")
	LINV_C_SP = scipy.sparse.triu(LINV_C, k=-nr, format="csr")

	print "Sparsified Inverted Matrices .... \n"

	return LINV_psi_SP,LINV_T_SP,LINV_C_SP;

#LINV_psi_SP,LINV_T_SP,LINV_C_SP  = Matrices(dt_newton);
#LINV_psi_SP,LINV_T_SP,LINV_C_SP  = Matrices(dt)

#@profile
def TimeStep(Iters,X,Ra):

	# 1) Time-step
	NX = np.zeros((3*N,1));
	X_Norm = []; Time =[]; X_DATA = [];

	start_time = time.time()
	RAG = np.zeros((N,1)); T0_PSI = np.zeros((N,1));

	err = 1.0;
	for ii in xrange(Iters):
		
		PSI = X[0:N]; T = X[N:2*N]; C = X[2*N:3*N];
		
		# Nonlinear,Speed Up
		NX[:,0] = dt*Non_lin(D,R,N_fm,kxS,aks,kxC,akc,JT,A2_SINE,X); # 56%

		# Linear
		RAG[:,0] = dt*Ra*gr.dot(X[N:2*N,0] - S*X[2*N:3*N,0])
		T0_PSI[:,0] = dt*T0_JT.dot(X[0:N,0]); # Add T'0*psi

		# 1) Vorticity
		NX[0:N] += A2.dot(X[0:N]) + RAG
		PSI_new = LINV_psi_SP.dot(NX[0:N]) # <12%

		# 2) Temperature
		NX[N:2*N] += Rsq.dot(X[N:2*N]) + T0_PSI
		T_new = LINV_T_SP.dot(NX[N:2*N]); # <12 ?%

		# 3) Concentration
		NX[2*N:3*N] += Rsq.dot(X[2*N:3*N]) + T0_PSI
		C_new = LINV_C_SP.dot(NX[2*N:3*N]); # 12%

		if ii%100 == 0:
			print "Iter ",ii
			err = np.linalg.norm(PSI_new - PSI,2)/np.linalg.norm(PSI_new,2)
			print "Error ",err,"\n"

		if ii%nsave == 0:	
			print "Appended Frame "
			X_DATA.append( np.vstack((PSI_new,T_new,C_new)) );

		X[0:N] = PSI_new; X[N:2*N] = T_new; X[2*N:3*N] = C_new;

		X_Norm.append(np.linalg.norm(X,2))
		Time.append(dt*ii)		


	end_time = time.time()
	print("Elapsed time, time-step, was %g seconds" % (end_time - start_time))	


	return X_Norm,Time,X_DATA;		

'''
# 1) Call time-stepper
X_Norm, Time,X_DATA = TimeStep(N_steps,X,Ra) 

# 2) Make Directory & Save Files, USE PANDAS
os.mkdir("".join(Stringy))
# 3) Change into Results folder
os.chdir("".join(Stringy))	

#Create an associated Parameters file
Parameters = np.zeros(18)
file = open("Parameters.txt",'w')

file.write(" #~~~~~~~~~~~~~~~~ Control Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

Parameters[0] = Ra_c
file.write('%5.4f'% Ra_c); file.write(" # Ra_c Critical Rayleigh #0 \n");

Parameters[1] = eps
file.write('%5.4f'% eps); file.write(" # epsilon #1 \n")

Parameters[2] = Tau
file.write('%5.4f'% Tau); file.write(" # Inverse Lewis #2 \n" )

Parameters[3] = Pr
file.write('%5.4f'% Pr); file.write(" # Pr Prandtl nunber #3\n");	

Parameters[4] = d
file.write('%5.3f'% d); file.write(" # d Sigma Gap Width #4 \n")

Parameters[5] = Ra_s
file.write('%5.4f'% Ra_s); file.write(" # Ra_s Solute Rayleigh #5 \n");

file.write(" #~~~~~~~~~~~~~~~~ Numerical Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

Parameters[6] = dt
file.write('%1.6f' % dt); file.write(" # Time step #6 \n")

Parameters[7] = Nr
file.write('%d' % Nr); file.write(" # Nr Radial Colocation #7 \n")

Parameters[8] = N_fm
file.write('%d' % N_fm);	file.write(" # N_fm Fourirt Modes #8 \n")

Parameters[9] = nsave
file.write('%d' % nsave);	file.write(" nsave #9 \n")


STRVID = "".join(['X_VID_l20.npy'])

np.save("X_Norm_L2.npy",X_Norm)
np.save("Time.npy",Time)
np.save(STRVID,X_DATA)
np.save("Parameters.npy",Parameters)

# Exit Directory & Plot
os.chdir('/home/mannixp/Dropbox/Leeds/')

plt.plot(Time,X_Norm,'k:')
plt.plot(Time,X_Norm,'bo',markersize=0.3)
plt.xlabel(r'$T$',fontsize=25)
plt.xlabel(r'$||X||$',fontsize=25)
plt.show()

X = X_DATA[-1]

sys.exit()
'''

#@profile
def Newton(X,Ra):
	
	start_time = time.time()

	NX = np.zeros(X.shape); FX = np.zeros(3*N);
	X_new = np.zeros(X.shape); dv = 1e-05*np.random.randn(3*N);
	
	err = 1.0; it = 0;
	while err > 1e-12:

		# 1) Return F(X) & Nonlinear Terms in Grid-Space
		NX[:,0], X_sol = Non_lin_WTerms(D,R,N_fm,aks,akc,JT,A2_SINE,X); # 56%
					
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~ 
		# Fx = P*(Lx + N(x,x))
		FX = PRECOND( LINV_psi_SP,LINV_T_SP,LINV_C_SP, LIN_LX(Ra*gr,T0_JT,A4,NABLA2,Tau,S, X) + NX )[:,0];
		
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~
		# DF*dv = P*(L*dv + N(x,dv) + N(dv,x))
		DF = lambda dv: PRECOND( LINV_psi_SP,LINV_T_SP,LINV_C_SP,  LIN_LX(Ra*gr,T0_JT,A4,NABLA2,Tau,S, dv) + LIN_FX(D,R,N_fm,aks,akc,JT,A2_SINE,X_sol, dv) );

		# Define Lin Operator
		DFX = spla.LinearOperator((3*N,3*N),matvec=DF,dtype='float64');


		print "Conj gradient solving ..."
		# Solve A(v)*dv = b(v); # Already pre-conditioned
		
		##start_time = time.time()
		#SOL = spla.bicgstab(DFX, FX, x0=dv, tol=1e-05, maxiter=50) #,M=cP_inv); # Faster no conv
		SOL = spla.gmres(DFX, FX, tol=1e-05, maxiter=25) #,M=cP_inv); # Slow but converegs
		##end_time = time.time()
		##print("Elapsed time, bicgstab, was %g seconds" % (end_time - start_time))

		dv = SOL[0]; X_new[:,0] = X[:,0] - dv;
		print "CONVERGED: INFO",SOL[1]
				
		err = np.linalg.norm(dv,2)/np.linalg.norm(X_new,2)							
		#if it%2 == 0:
		print 'Error ',err
		print 'Iteration: ', it,"\n"

		X = X_new;
		it+=1

	end_time = time.time()
	print("Elapsed time, Newton, was %g seconds" % (end_time - start_time))	

	return X;

#@profile
def Contin(Y):#,Y_dot):

	# Hyper-parameters
	delta = 0.1; # Too big doesn't follow it, Too small causes branch jumping
	sign = -1.; # decides which direction

	# Un-Pack Input
	X_0 = Y[0:-1]; mu_0 = Y[-1,0];

	#print " X_o shape ",X_0.shape
	#print " mu_0 ",mu_0,"\n"

	DIM = len(X_0); s = (DIM,DIM); sG = (DIM + 1,DIM + 1)
	X = np.zeros((DIM,1)); mu = 0.0
	
	# 1) ~~~~~~~~~~~~~~~~~ Compute Prediction ~~~~~~~~~~~~~~~~~ 
	F_mu = np.zeros((DIM,1)); NX = np.zeros((DIM,1))
	
	#if np.linalg.norm(Y_dot,2) == 0:
	F_mu[0:N,0] = LINV_psi_SP.dot( gr.dot(X_0[N:2*N,0] - S*X_0[2*N:3*N,0]) ); 

	# Multiplied gr by mu_0* = Ra*
	NX[:,0], X_sol = Non_lin_WTerms(D,R,N_fm,aks,akc,JT,A2_SINE,X_0);
	DF0 = lambda dv: PRECOND( LINV_psi_SP,LINV_T_SP,LINV_C_SP,  LIN_LX(mu_0*gr,T0_JT,A4,NABLA2,Tau,S, dv) + LIN_FX(D,R,N_fm,aks,akc,JT,A2_SINE,X_sol, dv) );
	
	## ADD WHILE LOOP HERE
	# 2) ~~~~~~~~~~~~~~~~~ Compute xi ~~~~~~~~~~~~~~~~~
	# a) Solve DF_u*xi = -F_mu	bicgstab
	DFx = spla.LinearOperator(s,matvec=DF0,dtype='float64');
	dv = np.random.randn(DIM,1);

	SOL = spla.gmres(DFx,-F_mu, tol=1e-05, maxiter=100);
	xi = SOL[0]; # Update xi
	print "xi conv ",SOL[1]
	#print "xi.shape ",xi.shape
	# b) ~~~~~~~~~~~~~~~~~ Compute x_dot, mu_dot algebra ~~~~~~~~~~
	mu_dot = sign/np.sqrt( 1.0 + delta*(np.linalg.norm(xi,2)-1.0) );
	X_dot = mu_dot*xi; # print "X_dot",X_dot.shape

	#else:
	#	X_dot = Y_dot[0:-1]; mu_dot = Y_dot[-1];	

	#print Y.shape
	#print X_dot.shape
	#print mu_dot	
	# c) ~~~~~~ Prediction ~~~~~~~~~~
	mu = mu_0 + mu_dot*ds; 
	X[:,0] = X_0[:,0] + X_dot*ds;
	Y[0:-1] = X; Y[-1,0] = mu; 

	dY = np.random.randn(DIM+1); G = np.zeros((DIM+1,1))
	Y_new = np.zeros((DIM+1,1))

	err = 1.0; it = 0; 
	while err > 1e-06:

		# 1) Return Preconditioned P*DF(X), P*F(X), P*DF_mu & Nonlinear Terms in Grid-Space
		NX[:,0], X_sol = Non_lin_WTerms(D,R,N_fm,aks,akc,JT,A2_SINE,X); # 56%

		# P*DF(X) Include mu*gr
		DF = lambda dv: PRECOND( LINV_psi_SP,LINV_T_SP,LINV_C_SP,  LIN_LX(mu*gr,T0_JT,A4,NABLA2,Tau,S, dv) + LIN_FX(D,R,N_fm,aks,akc,JT,A2_SINE,X_sol, dv) );
		# P*DF_mu*X
		F_mu[0:N,0] = LINV_psi_SP.dot( gr.dot(X[N:2*N,0] - S*X[2*N:3*N,0]) );
		
		# 2) Compute G
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~ 	
		# Fx = P*(Lx + N(x,x))
		G[0:-1,0] = PRECOND( LINV_psi_SP,LINV_T_SP,LINV_C_SP, LIN_LX(mu*gr,T0_JT,A4,NABLA2,Tau,S, X) + NX )[:,0]
		# p ~ arc-length condition
		G[-1,0] = delta*np.dot(np.transpose(X_dot),X - X_0) + (1.0-delta)*mu_dot*(mu - mu_0) - ds;
		
		# 3) COmpute DG
		#~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~ 
		#print "Shapes DF*DY ",DF(dY[0:-1]).shape
		#print "Shapes F_mu*DY ",(dY[-1]*F_mu).shape
		#print "Shapes X_dot*DY  ",delta*np.dot(np.transpose(X_dot),dY[0:-1]); #.shape
		#print "Shapes mu_dot*DY ",(1.0-delta)*mu_dot*dY[-1]

		DG = lambda dY: np.vstack( ( DF(dY[0:-1]) + F_mu*dY[-1],\
			delta*np.dot(np.transpose(X_dot),dY[0:-1]) + (1.0-delta)*mu_dot*dY[-1]) );
		
		DGy = spla.LinearOperator(sG,matvec=DG,dtype='float64');

		# 3) Solve, compute error and update
		SOL = spla.gmres(DGy,-G, tol=1e-05, maxiter=30); #Omitted Preconditioner ?
		dY = SOL[0]; Y_new[:,0] = Y[:,0] + dY; # Update Y
		

		# 4) ~~~~~~~~~~~~~~~~~ Compute error ~~~~~~~~~~~~~~~~~
		err = np.linalg.norm(dY,2)/np.linalg.norm(Y[:,0] + dY,2);
		if (it%4==0) and (it > 0):
			print "Inv G ",SOL[1]
			print 'Pseudo Error ',err
			print 'Corrct Iter: ', it
			print "mu Update  = ",Y[-1,0],"\n"

		Y = Y_new;
		X = Y[0:-1]; mu = Y[-1,0];
		it+=1;

		# 5)  ~~~~~~~~~~~~~~~~~  Arc-length control  ~~~~~~~~~~~~~~~~~ 
		if it > 10:
			it = 0;
			global ds
			if ds > 10.0:
				ds = 0.9*ds; # Smaller for better cusp following

				print "Repredicting: reduced ds ",ds,"\n"
				print "mu_dot",mu_dot

				X = X_0 + X_dot*ds;
				mu = mu_0 + mu_dot*ds;

				Y[0:-1] = X; Y[-1,0] = mu;

				dY = np.random.randn(DIM+1);

			elif ds < 1e-05:
				print "Break or Iterating More"
				# Add termination condition here
				ds = 1e-05;
		
		# 3) Step Length Control	

	# 1) Compute new tangent X_dot, mu_dot
	
	'''
	#dY = np.random.randn(DIM+1); 
	b = np.zeros((DIM+1,1)); b[-1,0] = -1
	SOL = spla.gmres(DGy,b, tol=1e-05, maxiter=100); #Omitted Preconditioner ?
	Y_dot = SOL[0];
	print "Y_dot shape ",Y_dot.shape
	print "Tangent conv ",SOL[1]
	'''

	# 2) Update step-size ds
	if it < 3:
		#print ds
		global ds
		if ds < 10.0:
			ds = 1.1*ds;
			print "Increased ds ",ds,"\n"
		elif ds > 10.0:
			ds = 10.0		

	return Y;#,Y_dot;

X1 = np.load("TEST_L2.npy")

#LINV_psi_SP,LINV_T_SP,LINV_C_SP  = Matrices(dt)

#X_Norm, Time,X_DATA = TimeStep(1000,X1,Ra) 

LINV_psi_SP,LINV_T_SP,LINV_C_SP  = Matrices(dt_newton); 

print "Netwon Iter \n"
Ra = Ra + 10.0;
X = Newton(X1,Ra);


# Call Continuation
mu_r = []; v_data = [];
#r_step = 1.0; ru = r; 
ds = 10.0; N = nr*N_fm;
Y = np.zeros( (3*N + 1,1) );
Y_dot = np.zeros( 3*N + 1 );
Y[0:-1] = X; Y[-1] = Ra;
for ii in xrange(10):

	#if ii%5 == 0:
	print "ITER ",ii
	print "Ra =",Y[-1,0],"\n"

	#v_new = Netwon_It(v,ru,ak);
	#Y_new,Y_dot_new = Contin(Y,Y_dot)
	Y_new = Contin(Y)
	v_data.append(np.linalg.norm(Y_new[0:-1],2));
	mu_r.append(Y_new[-1,0]);
	Y = Y_new; #Y_dot = Y_dot_new; 
	#ru = ru + r_step;


plt.plot(mu_r,v_data,'bo',markersize=1.5)
plt.plot(mu_r,v_data,'b-',linewidth=1.0)
plt.show()


sys.exit()


from Plot_Tools import *
Theta_grid = x;#np.linspace(0,np.pi,N_fm); 
r_grid = np.linspace(R[-1],R[0],50);
PSI, T, C,T_0 = SPEC_TO_GRID(R,r_grid,N_fm,X,d)
Plot_Package_CHE(r_grid,Theta_grid,C+T_0,PSI,T+T_0,d);#+T_0

Energy(X,N_fm,R);