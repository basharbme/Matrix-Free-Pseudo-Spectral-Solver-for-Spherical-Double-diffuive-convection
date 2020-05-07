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

# *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~
# 3) ~~~~~~~~ Linear Blocks & Operators ~~~~~~~~~~~
# *~~~~~~~~~~~~~~~,"\n"~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~

# Build \nabla^2 Operator
def Nabla2(D,r): 
	# r^2 T'' +2r T'
	A = np.zeros(D.shape)
	#D,r = cheb_radial(N,d); # Returns len N+1 matrix
	D2 = np.matmul( np.diag(r[:]**2) , np.matmul(D,D) );
	RD = np.matmul( np.diag(2.0*r[:]),D);
	A = D2 + RD

	# Leaving out the edges enforces the dircihlet b.c.s
	return A[1:-1,1:-1]

# Build \nabla^2\nabla^2 Operator
def Nabla4(D,r): 
	
	# D,r = cheb_radial(N,d); # Returns len N+1 matrix
	I = np.ones(len(r));
	r_i = r[-1]; r_o = r[0];
	b = -(r_i + r_o); c = r_i*r_o
	
	S = np.diag(I/((r**2)+b*r+c*I));
	S[0,0] = 0.0; S[-1,-1] = 0.0;

	# All matrices are for v!!
	D2 = np.matmul(D,D);
	D3 = np.matmul(D,D2);
	D4 = np.matmul(D2,D2);
	
	# Define \tilde{D^4} + Implement BC
	L4 = np.matmul( np.diag(r**2 + b*r + c*I), D4) + np.matmul( 4.0*np.diag(2.0*r + b*I),D3) + 12.0*D2;
	D4 = np.matmul(L4,S); # (d/dr)^4
	
	# Define \tilde{D^3} + Implement BC
	#L3 = np.matmul( np.diag(r[:]**2 + b*r[:] + c), D3) + np.matmul( 3.0*np.diag(2.0*r[:] + b),D2) + 6.0*D;
	#D3 = np.matmul( np.diag(4.0/r[:]), np.matmul(L3,S) ); # (4/r)*(d/dr)^3


	# Implement BC??
	#D2 = -2.0*l*(l+1.0)*np.matmul(np.diag(1.0/(r[:]**2)),D2); 
	#L4 = ( (l*(l+1.0))**2 - 2.0*l*(l+1.0) )*np.diag(1.0/(r[:]**4));

	#A = D4;# + D3;# + D2 + L4;

	return D4[1:-1,1:-1];


# Build D^3 Operator
def Diff3(D,r): 
	
	# D,r = cheb_radial(N,d); # Returns len N+1 matrix
	I = np.ones(len(r));
	r_i = r[-1]; r_o = r[0];
	b = -(r_i + r_o); c = r_i*r_o
	
	S = np.diag(I/((r**2)+b*r+c*I));
	S[0,0] = 0.0; S[-1,-1] = 0.0;

	# All matrices are for v!!
	D2 = np.matmul(D,D);
	D3 = np.matmul(D,D2);
	#D4 = np.matmul(D2,D2);
	
	# Define \tilde{D^4} + Implement BC
	#L4 = np.matmul( np.diag(r**2 + b*r + c*I), D4) + np.matmul( 4.0*np.diag(2.0*r + b*I),D3) + 12.0*D2;
	#D4 = np.matmul(L4,S); # (d/dr)^4
	
	# Define \tilde{D^3} + Implement BC
	L3 = np.matmul( np.diag(r[:]**2 + b*r[:] + c), D3) + np.matmul( 3.0*np.diag(2.0*r[:] + b),D2) + 6.0*D;
	D3 = np.matmul(L3,S) # (d/dr)^3


	# Implement BC??
	#D2 = -2.0*l*(l+1.0)*np.matmul(np.diag(1.0/(r[:]**2)),D2); 
	#L4 = ( (l*(l+1.0))**2 - 2.0*l*(l+1.0) )*np.diag(1.0/(r[:]**4));

	#A = D4;# + D3;# + D2 + L4;

	return D3; #\ [1:-1,1:-1]	

# ~~~~~ Full Nr x N_fm blocks ~~~~

# ~~~~~ NABLA2 Cosine~~~~
# LAP2_theta Cosine-Basis  = r^2 \nabla^2 + A^2_\theta
def A2_theta_C(R,N_fm): # No 1/R^2
	nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	IR = np.eye(nr);

	AT = [];
	for j in xrange(N_fm): # [0,N_Fm -1]
		AT_j = [];
		for k in xrange(N_fm): # [0,N_Fm -1]
			
			if (k == j):
				AT_j.append(-k*(k + 1.0)*IR)
			elif (k > j) and ( (k+j)%2 == 0 ):
				
				if j == 0: # Due to integral for a_0 being 1/pi not 2/pi
					AT_j.append(-k*IR); 
				elif k>0:
					AT_j.append(-k*2.0*IR)	
			else:
				AT_j.append(None);
		AT.append(AT_j);				 
	
	#return np.bmat(AT)#.todense(); #bmat(AT,format="csr").todense();
	return bmat(AT,format="csr");

def NABLA2_COS(D,R,N_fm): # Correct
	
	'''
	nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	LAP_2 = [];
	for j in xrange(N_fm): # [0,N_Fm -1]
		LAP2_j = [];
		for k in xrange(N_fm): # [0,N_Fm -1]
			if (k == j) and (k > 0):
				LAP2_j.append(Nabla2(D,R));
			elif (k == j) and (k == 0):
				LAP2_j.append(2.0*Nabla2(D,R)); # Due to integral
			else:
				LAP2_j.append(Z0);
		LAP_2.append(LAP2_j);

	LAP2 = np.bmat(LAP_2);
	'''
	#nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	LAP_2 = [];
	DR = Nabla2(D,R); # r^2 T'' +2r T'
	for k in xrange(N_fm): # [0,N_Fm -1]
		LAP_2.append(DR);
		
	D2 = block_diag(LAP_2,format="csr");#.todense()
	AT = A2_theta_C(R,N_fm);

	return D2 + AT;

# ~~~~~ NABLA4 Sine~~~~
# LAP2_theta Sine-Basis, D^2 + (A^2_\theta)/r^2
def A2_theta_S(R,N_fm): # Has 1/R^2
	
	
	#nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	#Z0 = 0.0*eye(nr)
	#IR = diags( 1.0/(R[1:-1]**2),0,format="csr"); 

	IR = np.diag( 1.0/(R[1:-1]**2)); 
	
	AT = [];
	for jj in xrange(N_fm): 
		j = jj + 1; # Sine [1,N_Fm]
		
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # Sine [1,N_Fm]

			if (k == j):
				
				AT_j.append(-j*(j + 1.0)*IR);

			elif (k > j) and ( (k + j)%2 == 0 ):
				
				AT_j.append(-2.0*j*IR);

			else:
				AT_j.append(None);
		AT.append(AT_j);				 
	
	return bmat(AT,format="csr");
	'''

	nr = len(R[1:-1]); I = eye(nr); Z0 = 0.0*I
	IR = diags( 1.0/(R[1:-1]**2),0 ); AT = [];
	
	for jj in xrange(N_fm): 
		j = jj + 1; # Sine [1,N_Fm]
		
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # Sine [1,N_Fm]

			if (k == j):
				
				AT_j.append(-j*(j + 1.0)*IR);

			elif (k > j) and ( (k + j)%2 == 0 ):
				
				AT_j.append(-2.0*j*IR);

			else:
				AT_j.append(Z0);
		AT.append(AT_j);				 
	
	return bmat(AT);'''

def NABLA2_SINE(D,R,N_fm): # Correct
	
	'''
	nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	LAP_2 = [];
	for j in xrange(N_fm): # [1,N_Fm -1]
		LAP2_j = [];
		for k in xrange(N_fm): # [1,N_Fm -1]
			if (k == j):
				LAP2_j.append(Nabla2(D,R));
			else:
				LAP2_j.append(Z0);
		LAP_2.append(LAP2_j);

	LAP2 = np.bmat(LAP_2);
	'''

	LAP_2 = []; DR = np.matmul(D,D)[1:-1,1:-1]
	for j in xrange(N_fm): # [1,N_Fm -1]
		LAP_2.append(DR);
	D2 = block_diag(LAP_2,format="csr");#.todense()
	
	return D2 + A2_theta_S(R,N_fm);#.todense();

# LAP4 full Sine-Basis, includes 1/r^2
def NABLA4(D,R,N_fm):
		
	nr = len(R[1:-1]); #Z0 = np.zeros((nr,nr));
	IR2 = np.diag( 1.0/(R**2) ); IR  = np.diag(1.0/R); # Keep dense
	
	LAP_4 = []; DT = []; 
	D4 = Nabla4(D,R); 
	D2 = 2.0*np.matmul(D,D) - 4.0*np.matmul(IR,D) + 6.0*IR2; 
	for k in xrange(N_fm): # Sine [1,N_Fm]
		LAP_4.append(D4);
		DT.append(D2[1:-1,1:-1]);

	LAP4 = block_diag(LAP_4,format="csr").todense()
	DTT = block_diag(DT,format="csr").todense()
	A2_theta = A2_theta_S(R,N_fm).todense();

	return LAP4 + np.matmul(A2_theta,DTT) + np.matmul(A2_theta,A2_theta); 


# LAP4 full Sine-Basis, includes 1/r^
def D3_SINE_NL(D,R,N_fm):

	I = np.ones(len(R));
	IR  = np.diag(I/R); # Keep dense
	IR2 = np.diag(I/(R[:]**2) );
	#IR3 = np.diag(1.0/(R**4) );

	LAP_3 = []; DT = []; 
	
	D3 = Diff3(D,R); # In full format
	NAB1 = np.matmul(IR2, D3 - 2.*np.matmul(IR,np.matmul(D,D)) )[1:-1,1:-1];
	NAB2 = np.matmul(IR2, D -4.*IR)[1:-1,1:-1];
	for k in xrange(N_fm): # Sine [1,N_Fm]
		LAP_3.append(NAB1);
		DT.append(NAB2);

	LAP3 = block_diag(LAP_3,format="csr");#.todense()
	DTT = block_diag(DT,format="csr");#.todense()
	A2_theta = A2_theta_S(R,N_fm);#.todense(); # Includes 1/r^2

	#A1 = A2_theta.dot(DTT)
	#B1 = np.matmul(A2_theta.todense(),DTT.todense() )
	#print "Checks ",np.allclose(A1.todense(),B1,rtol=1e-15)

	return LAP3 + A2_theta.dot(DTT)

# ~~~~~ J_theta Cosine-Basis~~~~
# Includes -T'_0; #/r^2
def T0J_theta(R,N_fm,d): # Correct ?
	#nr = len(R[1:-1]); #Z0 = 0.0*np.eye(nr); #np.zeros((nr,nr));
	A_t = -(1.0+d)/d; IR = -A_t*diags( 1.0/(R[1:-1]**2),0,format="csr"); 
	#IR = eye(nr);

	JT = [];
	for j in xrange(N_fm): # j cosine [0,N_Fm -1]
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # k Sine [1,N_Fm]
			
			if (k == j) and (k > 0): #?
				AT_j.append( (k + 1.0)*IR); 
			elif (k > j) and ( (j+k)%2 == 0 ):
				if j == 0:
					AT_j.append(IR);
				else:
					AT_j.append(2.*IR);

				#AT_j.append(2.0*IR);
			else:
				AT_j.append(None);
		
		JT.append(AT_j);				 
			
	return bmat(JT,format="csr");

# ~~~~~ g(r)_d_theta ~~~~
def kGR(R,N_fm): # Correct
	
	nr = len(R[1:-1]); 
	'''Z0 = np.zeros((nr,nr));
	GR = np.diag( 1.0/(R[1:-1]**2) ) 

	AT = [];
	for jj in xrange(N_fm): 
		j = jj + 1; # j Sine [1,N_Fm]
		
		AT_j = [];
		for k in xrange(N_fm): # k Cosine [0,N_Fm-1] 
			if (k == j):
				AT_j.append(-k*GR); 
			else:
				AT_j.append(Z0);
		
		AT.append(AT_j);				 
	
	A= np.bmat(AT)'''

	GR = diags( 1.0/(R[1:-1]**2), 0 ,format="csr") 
	
	AT = [];
	for jj in xrange(N_fm): 
		j = jj + 1; # j Sine [1,N_Fm]
		for k in xrange(N_fm): # k Cosine [0,N_Fm-1] 
			if (k == j):
				AT.append(-k*GR);
	
	A1 = block_diag(AT,format="csr");

	ID = 0.*identity(nr*N_fm,format="csr");
	ID[:-1*nr,1*nr:] = A1; # FIX V slow here

	return ID;


# Non linear Terms
# Excludes radial
def J_theta(nr,N_fm): # Correct
	
	I = eye(nr); JT = [];
	for j in xrange(N_fm): # j cosine [0,N_Fm -1]
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # k Sine [1,N_Fm]
			
			if (k == j) and (k > 0):
				AT_j.append( (k + 1.0)*I); 
			elif (k > j) and ( (j+k)%2 == 0 ):
				
				if j == 0:
					AT_j.append(I);
				else:
					AT_j.append(2.*I);
			else:
				AT_j.append(None);
		
		JT.append(AT_j);				 
			
	return bmat(JT,format="csr");#.todense();

# ~~~~~ NABLA4 Sine~~~~
# LAP2_theta Sine-Basis, ( D^2 + A^2/r^2)/r^2
def NL_A2_theta_S(R,N_fm): # Correct
	
	
	#nr = len(R[1:-1]); Z0 = np.zeros((nr,nr));
	#Z0 = 0.0*eye(nr)
	IR = diags( 1.0/(R[1:-1]**4),0,format="csr"); 
	#IR = np.diag( 1.0/(R[1:-1]**4)); 

	AT = [];
	for jj in xrange(N_fm): 
		j = jj + 1; # Sine [1,N_Fm]
		
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # Sine [1,N_Fm]

			if (k == j):
				
				AT_j.append(-j*(j + 1.0)*IR);

			elif (k > j) and ( (k + j)%2 == 0 ):
				
				AT_j.append(-2.0*j*IR);

			else:
				AT_j.append(None);
		AT.append(AT_j);				 
	
	return bmat(AT,format="csr");#.todense();
	'''

	nr = len(R[1:-1]); I = eye(nr); Z0 = 0.0*I
	IR = diags( 1.0/(R[1:-1]**2),0 ); AT = [];
	
	for jj in xrange(N_fm): 
		j = jj + 1; # Sine [1,N_Fm]
		
		AT_j = [];
		for kk in xrange(N_fm): 
			k = kk + 1; # Sine [1,N_Fm]

			if (k == j):
				
				AT_j.append(-j*(j + 1.0)*IR);

			elif (k > j) and ( (k + j)%2 == 0 ):
				
				AT_j.append(-2.0*j*IR);

			else:
				AT_j.append(Z0);
		AT.append(AT_j);				 
	
	return bmat(AT);'''

def NABLA2_SINE_NL(D,R,N_fm): # Correct
	IR = np.diag( (1.0/R**2) ); 

	LAP_2 = []; DR = np.matmul( IR,np.matmul(D,D) )[1:-1,1:-1]
	for j in xrange(N_fm): # [1,N_Fm -1]
		LAP_2.append(DR);
	D2 = block_diag(LAP_2,format="csr");#.todense()
	return D2 + NL_A2_theta_S(R,N_fm);

#@profile

# ~~~~~~~~ Full Operators  ~~~~~~~~~~~~
def M_0(D,R,N_fm): # Correct

	I = eye(nr*N_fm)
	A2 = NABLA2_SINE(D,R,N_fm); # Expensive
	return block_diag((A2,I,I),format="csc")

def L_Blocks(D,R,N_fm,d): #Sub Blocks

	A4 = NABLA4(D,R,N_fm);
	gr = kGR(R,N_fm);
	
	# Line 2
	J_T = T0J_theta(R,N_fm,d);
	A2 = NABLA2_COS(D,R,N_fm);

	return A4,gr,J_T,A2;

def L_X(A4,RaG,J_T,A2,Pr,Ra,Tau,X): # Block Action
	
	N =  len(X)/3;
	psi = X[0:N]; T = X[N:2*N]; C = X[2*N:3*N];

	# Line 1
	X[0:N] = Pr*A4.dot(psi) + Pr*Ra*RaG.dot(T-C); # Expensive as dense TU
	
	# Line 2
	JT_psi = J_T.dot(psi);
	X[N:2*N] = JT_psi + A2.dot(T)

	# Line 3
	X[2*N:3*N] = JT_psi + Tau*A2.dot(C)
	
	return X; # L*X

def L_0(D,R,N_fm,d,Pr,Ra,Tau): # Full block

	A4 = Pr*NABLA4(D,R,N_fm);
	#gr = Pr*Ra*kGR(R,N_fm);
	
	# Line 2
	#J_T = T0J_theta(R,N_fm,d);
	A2 = NABLA2_COS(D,R,N_fm);

	return block_diag((A4,A2,Tau*A2),format="csc")
	#return bmat([[A4,gr,-gr],[J_T,A2,None],[J_T,None,Tau*A2]],format='csc')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

# *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~





# *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~
# 4) ~~~~~~~~ Non-Linear Term ~~~~~~~~~~~
# *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~

#@profile
#~~~~~~~~~~~ # ~~~~~~~~~ Time-stepping ~~~~~~~~~~~~~
def Non_lin(D,R,N_fm,kxS,aks,kxC,akc,J_theta,A2_SINE,X): #,D3_SINE

	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	Dr = D[1:-1,1:-1];
	#IR = diags( 1.0/(R[1:-1]**2),0,format="csr"); 

	# 1) Compute derivatives & Transform to Nr x N_fm
	nr = len(R[1:-1]); N = nr*N_fm; sp = (nr,3*(N_fm/2)); # ZERO-PADDED ALIASING !!!!!
	#~~~~~~~~~~~ Wave-number Space ~~~~~~~~~~~~~~

	# DCT's
	JT_psi_hat = np.zeros(sp); DJT_psi_hat = np.zeros(sp); Domega_hat = np.zeros(sp); 
	komega_hat = np.zeros(sp);  kDpsi_hat = np.zeros(sp); DT_hat = np.zeros(sp); DC_hat = np.zeros(sp); 
	
	# DST's
	omega_hat = np.zeros(sp);  Dpsi_hat = np.zeros(sp); kT_hat = np.zeros(sp); kC_hat = np.zeros(sp);

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	
	# Use dot as-sparse matrices
	#JPSI = np.dot(J_theta,X[0:N]); 
	#OMEGA = np.dot(A2_SINE, X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = np.dot(D3_SINE, X[0:N]);
	
	JPSI = J_theta.dot(X[0:N]); 
	OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = D3_SINE.dot(X[0:N]);

	
	# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm 
	# O(nr^2*N_fm)
	for ii in xrange(N_fm):

		# Wavenumbers
		k_s = ii + 1; # [1,N_fm]
		k_c = ii; # [0,N_fm-1]
		
		# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ ???
		ind_p = ii*nr; psi = X[ind_p:ind_p+nr,0];

		Dpsi_hat[:,ii] = np.matmul(Dr,psi);
		kDpsi_hat[:,ii] = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine #
				
		JT_psi_hat[:,ii] = JPSI[ind_p:ind_p+nr,0];
		DJT_psi_hat[:,ii] =  np.matmul(Dr,JT_psi_hat[:,ii]) 

		omega_hat[:,ii] = OMEGA[ind_p:ind_p+nr,0];
		komega_hat[:,ii] = k_s*omega_hat[:,ii] # Sine -> Cosine 

		Domega_hat[:,ii] = np.matmul(Dr,omega_hat[:,ii]); 
		#Domega_hat[:,ii] = DOMEGA[ind_p:ind_p+nr,0];


		# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
		ind_T = N + ii*nr; T = X[ind_T:ind_T+nr,0];

		DT_hat[:,ii] = np.matmul(Dr,T);
		kT_hat[:,ii] = -k_c*T; # Cosine -> Sine

		# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
		ind_C = 2*N + ii*nr; C = X[ind_C:ind_C+nr,0];

		DC_hat[:,ii] = np.matmul(Dr,C);
		kC_hat[:,ii] = -k_c*C; # Cosine -> Sine

	
	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;

	# b) sine -> cosine
	kDpsi_hat[:,1:] = kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;
	

	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# ~~~~~~~~~~ Physical Space ~~~~~~~~~~~
	# DCT's
	JT_psi = np.zeros(sp); DJT_psi = np.zeros(sp); Domega = np.zeros(sp); 
	komega = np.zeros(sp);  kDpsi = np.zeros(sp); DT = np.zeros(sp); DC = np.zeros(sp); 
	
	# DST's
	omega = np.zeros(sp);  Dpsi = np.zeros(sp); kT = np.zeros(sp); kC = np.zeros(sp);
	#~~~~~~~~~~~~ #~~~~~~~~~~~~~~ #~~~~~~~~~~~~~

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) # Correct

		# a) ~~~ iDCT ~~~~
		# a) psi parts
		DJT_psi[jj,:] = idct(DJT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		JT_psi[jj,:] = idct(JT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		

		#w = np.roll(kDpsi[jj,:],1); w[0] = 0.0;
		kDpsi[jj,:] = idct(kDpsi_hat[jj,:],type=2,norm='ortho') # needs shift
		
		komega[jj,:] = idct(komega_hat[jj,:],type=2,norm='ortho') # needs shift
		
		# b) T parts
		DT[jj,:] = idct(DT_hat[jj,:],type=2,norm='ortho') # Projected okay
		# c) C parts
		DC[jj,:] = idct(DC_hat[jj,:],type=2,norm='ortho') # Projected okay


		# b) ~~~ iDST ~~~~	
		# a) psi parts
		omega[jj,:] = idst(omega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Domega[jj,:] = idst(Domega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Dpsi[jj,:] = idst(Dpsi_hat[jj,:],type=2,norm='ortho') # Projected okay

		# Cosine -> Sine, shift back with u[-1] = 0.0
		# b) T parts
		kT[jj,:] = idst(kT_hat[jj,:],type=2,norm='ortho') # Needs shift
		
		# c) C parts
		kC[jj,:] = idst(kC_hat[jj,:],type=2,norm='ortho') # Needs shift
		
	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NJ_PSI = DJT_psi*omega + JT_psi*Domega - (kDpsi*omega + Dpsi*komega);

	NJ_PSI_T = JT_psi*DT - Dpsi*kT;
	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;


	# 4) Compute DCT and DST, un-pad, multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	s = (nr,N_fm); # Un-PAD WITH ZEROS FOR De-ALIASING !!!!!
	J_PSI_hat = np.zeros(s); J_PSI_T_hat = np.zeros(s); J_PSI_C_hat = np.zeros(s); 

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) Correct
		J_PSI_hat[jj,:] = aks*( dst(NJ_PSI[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_T_hat[jj,:] =	akc*( dct(NJ_PSI_T[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_C_hat[jj,:] = akc*( dct(NJ_PSI_C[jj,:],type=2,norm='ortho')[0:N_fm] ); 

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NX = np.zeros(3*N)

	for ii in xrange(N_fm): # O(N_fm) Correct
		# a) psi parts
		ind_p = ii*nr;
		NX[ind_p:ind_p+nr] = J_PSI_hat[:,ii];
		# b) T parts
		ind_T = N + ii*nr; 
		NX[ind_T:ind_T+nr] = J_PSI_T_hat[:,ii]
		# c) C parts
		ind_C = 2*N + ii*nr;
		NX[ind_C:ind_C+nr] = J_PSI_C_hat[:,ii]#IR.dot(J_PSI_C_hat[:,ii]) # Multip 1/r^2

	return -NX; # (-1) As we write F(X) = N(X,X) + L*X;

#~~~~~~~~~~~ # ~~~~~~~~~ Netwon Iteration ~~~~~~~~~~~~~
#@profile
def Non_lin_WTerms(D,R,N_fm,aks,akc,J_theta,A2_SINE,X):

	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	Dr = D[1:-1,1:-1];
	#IR = diags( 1.0/(R[1:-1]**2),0,format="csr"); 

	# 1) Compute derivatives & Transform to Nr x N_fm
	nr = len(R[1:-1]); N = nr*N_fm; sp = (nr,3*(N_fm/2)); # ZERO-PADDED ALIASING !!!!!
	#~~~~~~~~~~~ Wave-number Space ~~~~~~~~~~~~~~

	# DCT's
	JT_psi_hat = np.zeros(sp); DJT_psi_hat = np.zeros(sp); Domega_hat = np.zeros(sp); 
	komega_hat = np.zeros(sp);  kDpsi_hat = np.zeros(sp); DT_hat = np.zeros(sp); DC_hat = np.zeros(sp); 
	
	# DST's
	omega_hat = np.zeros(sp);  Dpsi_hat = np.zeros(sp); kT_hat = np.zeros(sp); kC_hat = np.zeros(sp);

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	
	# Use dot as-sparse matrices
	#JPSI = np.dot(J_theta,X[0:N]); 
	#OMEGA = np.dot(A2_SINE, X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = np.dot(D3_SINE, X[0:N]);
	
	JPSI = J_theta.dot(X[0:N]); 
	OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = D3_SINE.dot(X[0:N]);

	
	# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm 
	# O(nr^2*N_fm)
	for ii in xrange(N_fm):

		# Wavenumbers
		k_s = ii + 1; # [1,N_fm]
		k_c = ii; # [0,N_fm-1]
		
		# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ ???
		ind_p = ii*nr; psi = X[ind_p:ind_p+nr,0];

		Dpsi_hat[:,ii] = np.matmul(Dr,psi);
		kDpsi_hat[:,ii] = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine #
				
		JT_psi_hat[:,ii] = JPSI[ind_p:ind_p+nr,0];
		DJT_psi_hat[:,ii] =  np.matmul(Dr,JT_psi_hat[:,ii]) 

		omega_hat[:,ii] = OMEGA[ind_p:ind_p+nr,0];
		komega_hat[:,ii] = k_s*omega_hat[:,ii] # Sine -> Cosine 

		Domega_hat[:,ii] = np.matmul(Dr,omega_hat[:,ii]); 
		#Domega_hat[:,ii] = DOMEGA[ind_p:ind_p+nr,0];


		# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
		ind_T = N + ii*nr; T = X[ind_T:ind_T+nr,0];

		DT_hat[:,ii] = np.matmul(Dr,T);
		kT_hat[:,ii] = -k_c*T; # Cosine -> Sine

		# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
		ind_C = 2*N + ii*nr; C = X[ind_C:ind_C+nr,0];

		DC_hat[:,ii] = np.matmul(Dr,C);
		kC_hat[:,ii] = -k_c*C; # Cosine -> Sine

	
	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;

	# b) sine -> cosine
	kDpsi_hat[:,1:] = kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;
	

	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# ~~~~~~~~~~ Physical Space ~~~~~~~~~~~
	# DCT's
	JT_psi = np.zeros(sp); DJT_psi = np.zeros(sp); Domega = np.zeros(sp); 
	komega = np.zeros(sp);  kDpsi = np.zeros(sp); DT = np.zeros(sp); DC = np.zeros(sp); 
	
	# DST's
	omega = np.zeros(sp);  Dpsi = np.zeros(sp); kT = np.zeros(sp); kC = np.zeros(sp);
	#~~~~~~~~~~~~ #~~~~~~~~~~~~~~ #~~~~~~~~~~~~~

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) # Correct

		# a) ~~~ iDCT ~~~~
		# a) psi parts
		DJT_psi[jj,:] = idct(DJT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		JT_psi[jj,:] = idct(JT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		

		#w = np.roll(kDpsi[jj,:],1); w[0] = 0.0;
		kDpsi[jj,:] = idct(kDpsi_hat[jj,:],type=2,norm='ortho') # needs shift
		
		komega[jj,:] = idct(komega_hat[jj,:],type=2,norm='ortho') # needs shift
		
		# b) T parts
		DT[jj,:] = idct(DT_hat[jj,:],type=2,norm='ortho') # Projected okay
		# c) C parts
		DC[jj,:] = idct(DC_hat[jj,:],type=2,norm='ortho') # Projected okay


		# b) ~~~ iDST ~~~~	
		# a) psi parts
		omega[jj,:] = idst(omega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Domega[jj,:] = idst(Domega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Dpsi[jj,:] = idst(Dpsi_hat[jj,:],type=2,norm='ortho') # Projected okay

		# Cosine -> Sine, shift back with u[-1] = 0.0
		# b) T parts
		kT[jj,:] = idst(kT_hat[jj,:],type=2,norm='ortho') # Needs shift
		
		# c) C parts
		kC[jj,:] = idst(kC_hat[jj,:],type=2,norm='ortho') # Needs shift
		
	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NJ_PSI = DJT_psi*omega + JT_psi*Domega - (kDpsi*omega + Dpsi*komega);

	NJ_PSI_T = JT_psi*DT - Dpsi*kT;
	
	NJ_PSI_C = JT_psi*DC - Dpsi*kC;


	# 4) Compute DCT and DST, un-pad, multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	s = (nr,N_fm); # Un-PAD WITH ZEROS FOR De-ALIASING !!!!!
	J_PSI_hat = np.zeros(s); J_PSI_T_hat = np.zeros(s); J_PSI_C_hat = np.zeros(s); 

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) Correct
		J_PSI_hat[jj,:] = aks*( dst(NJ_PSI[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_T_hat[jj,:] =	akc*( dct(NJ_PSI_T[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_C_hat[jj,:] = akc*( dct(NJ_PSI_C[jj,:],type=2,norm='ortho')[0:N_fm] ); 

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NX = np.zeros(3*N)

	for ii in xrange(N_fm): # O(N_fm) Correct
		# a) psi parts
		ind_p = ii*nr;
		NX[ind_p:ind_p+nr] = J_PSI_hat[:,ii];
		# b) T parts
		ind_T = N + ii*nr; 
		NX[ind_T:ind_T+nr] = J_PSI_T_hat[:,ii]
		# c) C parts
		ind_C = 2*N + ii*nr;
		NX[ind_C:ind_C+nr] = J_PSI_C_hat[:,ii]#IR.dot(J_PSI_C_hat[:,ii]) # Multip 1/r^2

	# Pack computed terms DIM (nr,3/2*N_fm)
	X_sol = [JT_psi,DJT_psi,Dpsi,kDpsi,omega,Domega,komega,DT,DC,kT,kC];	
	
	return -NX, X_sol;

#@profile
def LIN_FX(D,R,N_fm,aks,akc,J_theta,A2_SINE,X_sol,X_in):

	# Dr must be D[1:-1,1:-1]
	# A2 must be sine
	# J_theta must have no r dependancy
	
	Dr = D[1:-1,1:-1];
	#IR = diags( 1.0/(R[1:-1]**2),0,format="csr"); 

	# 1) Compute derivatives & Transform to Nr x N_fm
	nr = len(R[1:-1]); N = nr*N_fm; sp = (nr,3*(N_fm/2)); # ZERO-PADDED ALIASING !!!!!
	#~~~~~~~~~~~ Wave-number Space ~~~~~~~~~~~~~~
	s = (3*N,1)
	if X_in.shape != s:
		#print "Cast X_in to shape ",X_in.shape
		X = np.zeros((3*N,1)); X[:,0] = X_in;
		#print "Casted X to shape ",X.shape
	else:
		X = X_in	

	# DCT's
	JT_psi_hat = np.zeros(sp); DJT_psi_hat = np.zeros(sp); Domega_hat = np.zeros(sp); 
	komega_hat = np.zeros(sp);  kDpsi_hat = np.zeros(sp); DT_hat = np.zeros(sp); DC_hat = np.zeros(sp); 
	
	# DST's
	omega_hat = np.zeros(sp);  Dpsi_hat = np.zeros(sp); kT_hat = np.zeros(sp); kC_hat = np.zeros(sp);

	# length N vector + Perform theta derivatives O( (nr*N_fm )^2 )
	
	# Use dot as-sparse matrices
	#JPSI = np.dot(J_theta,X[0:N]); 
	#OMEGA = np.dot(A2_SINE, X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = np.dot(D3_SINE, X[0:N]);
	
	JPSI = J_theta.dot(X[0:N]); 
	OMEGA = A2_SINE.dot(X[0:N]); # Check Yields (A^2 \psi)/r^2 and enfores \psi = 0
	#DOMEGA = D3_SINE.dot(X[0:N]);

	
	# Take Radial Deriv, Reshape ; nr*N_fm -> nr x N_fm 
	# O(nr^2*N_fm)
	for ii in xrange(N_fm):

		# Wavenumbers
		k_s = ii + 1; # [1,N_fm]
		k_c = ii; # [0,N_fm-1]
		
		# a) ~~~~~~~ psi parts ~~~~~~~~~~~~ ???
		ind_p = ii*nr; psi = X[ind_p:ind_p+nr,0];

		Dpsi_hat[:,ii] = np.matmul(Dr,psi);
		kDpsi_hat[:,ii] = k_s*Dpsi_hat[:,ii]; # Sine -> Cosine #
				
		JT_psi_hat[:,ii] = JPSI[ind_p:ind_p+nr,0];
		DJT_psi_hat[:,ii] =  np.matmul(Dr,JT_psi_hat[:,ii]) 

		omega_hat[:,ii] = OMEGA[ind_p:ind_p+nr,0];
		komega_hat[:,ii] = k_s*omega_hat[:,ii] # Sine -> Cosine 

		Domega_hat[:,ii] = np.matmul(Dr,omega_hat[:,ii]); 
		#Domega_hat[:,ii] = DOMEGA[ind_p:ind_p+nr,0];


		# b) ~~~~~~~~~~ T parts ~~~~~~~~~~~~~ # Correct
		ind_T = N + ii*nr; T = X[ind_T:ind_T+nr,0];

		DT_hat[:,ii] = np.matmul(Dr,T);
		kT_hat[:,ii] = -k_c*T; # Cosine -> Sine

		# c) ~~~~~~~~~~ C parts ~~~~~~~~~~~~ # Correct
		ind_C = 2*N + ii*nr; C = X[ind_C:ind_C+nr,0];

		DC_hat[:,ii] = np.matmul(Dr,C);
		kC_hat[:,ii] = -k_c*C; # Cosine -> Sine

	
	# Preform all rolling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# a)cosine -> sine
	kT_hat[:,0:-1] = kT_hat[:,1:]; kT_hat[:,-1] = 0.0;
	kC_hat[:,0:-1] = kC_hat[:,1:]; kC_hat[:,-1] = 0.0;

	# b) sine -> cosine
	kDpsi_hat[:,1:] = kDpsi_hat[:,0:-1]; kDpsi_hat[:,0] = 0.0;
	komega_hat[:,1:] = komega_hat[:,0:-1]; komega_hat[:,0] = 0.0;
	

	# 2) ~~~~ Compute iDCT & iDST ~~~~~ # 
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~

	# ~~~~~~~~~~ Physical Space ~~~~~~~~~~~
	# DCT's
	JT_psi = np.zeros(sp); DJT_psi = np.zeros(sp); Domega = np.zeros(sp); 
	komega = np.zeros(sp);  kDpsi = np.zeros(sp); DT = np.zeros(sp); DC = np.zeros(sp); 
	
	# DST's
	omega = np.zeros(sp);  Dpsi = np.zeros(sp); kT = np.zeros(sp); kC = np.zeros(sp);
	#~~~~~~~~~~~~ #~~~~~~~~~~~~~~ #~~~~~~~~~~~~~

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) # Correct

		# a) ~~~ iDCT ~~~~
		# a) psi parts
		DJT_psi[jj,:] = idct(DJT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		JT_psi[jj,:] = idct(JT_psi_hat[jj,:],type=2,norm='ortho') # Projected okay
		

		#w = np.roll(kDpsi[jj,:],1); w[0] = 0.0;
		kDpsi[jj,:] = idct(kDpsi_hat[jj,:],type=2,norm='ortho') # needs shift
		
		komega[jj,:] = idct(komega_hat[jj,:],type=2,norm='ortho') # needs shift
		
		# b) T parts
		DT[jj,:] = idct(DT_hat[jj,:],type=2,norm='ortho') # Projected okay
		# c) C parts
		DC[jj,:] = idct(DC_hat[jj,:],type=2,norm='ortho') # Projected okay


		# b) ~~~ iDST ~~~~	
		# a) psi parts
		omega[jj,:] = idst(omega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Domega[jj,:] = idst(Domega_hat[jj,:],type=2,norm='ortho') # Projected okay
		Dpsi[jj,:] = idst(Dpsi_hat[jj,:],type=2,norm='ortho') # Projected okay

		# Cosine -> Sine, shift back with u[-1] = 0.0
		# b) T parts
		kT[jj,:] = idst(kT_hat[jj,:],type=2,norm='ortho') # Needs shift
		
		# c) C parts
		kC[jj,:] = idst(kC_hat[jj,:],type=2,norm='ortho') # Needs shift
		
	
	# Compute all primed terms !!	

	# Un- Pack computed terms DIM (nr,3/2*N_fm)
	# X_sol = [JT_psi,DJT_psi,Dpsi,kDpsi,omega,Domega,komega,DT,DC,kT,kC];	
	JT_psi_b = X_sol[0]; DJT_psi_b = X_sol[1]; Dpsi_b = X_sol[2]; kDpsi_b = X_sol[3];
	omega_b = X_sol[4]; Domega_b = X_sol[5]; komega_b = X_sol[6];
	DT_b = X_sol[7]; DC_b = X_sol[8]; kT_b = X_sol[9]; kC_b = X_sol[10];

	# 3) Perform mulitplications in physical space O( (nr*N_fm)**2) Correct
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	# Calc N(X,X') using _b for base state
	NJ_PSI = DJT_psi_b*omega + JT_psi_b*Domega - (kDpsi_b*omega + Dpsi_b*komega);

	NJ_PSI_T = JT_psi_b*DT - Dpsi_b*kT;
	
	NJ_PSI_C = JT_psi_b*DC - Dpsi_b*kC;

	# Calc N(X',X) & add it to N(X,X')
	NJ_PSI += DJT_psi*omega_b + JT_psi*Domega_b - (kDpsi*omega_b + Dpsi*komega_b);

	NJ_PSI_T += JT_psi*DT_b - Dpsi*kT_b;
	
	NJ_PSI_C += JT_psi*DC_b - Dpsi*kC_b;


	# 4) Compute DCT and DST, un-pad, multiply by scaling factor aks,akc
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	s = (nr,N_fm); # Un-PAD WITH ZEROS FOR De-ALIASING !!!!!
	J_PSI_hat = np.zeros(s); J_PSI_T_hat = np.zeros(s); J_PSI_C_hat = np.zeros(s); 

	for jj in xrange(nr): # nr*O(N_fm*ln(N_fm)) Correct
		J_PSI_hat[jj,:] = aks*( dst(NJ_PSI[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_T_hat[jj,:] =	akc*( dct(NJ_PSI_T[jj,:],type=2,norm='ortho')[0:N_fm] ); 
		J_PSI_C_hat[jj,:] = akc*( dct(NJ_PSI_C[jj,:],type=2,norm='ortho')[0:N_fm] ); 

	# 5) Reshape ; 3 x Nr x N_fm -> 3*nr*N_fm ; Fill into NX
	# *~~~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~~~~ * ~~~~~~~~~
	NX = np.zeros( (3*N,1))

	for ii in xrange(N_fm): # O(N_fm) Correct
		# a) psi parts
		ind_p = ii*nr;
		NX[ind_p:ind_p+nr,0] = J_PSI_hat[:,ii];
		# b) T parts
		ind_T = N + ii*nr; 
		NX[ind_T:ind_T+nr,0] = J_PSI_T_hat[:,ii]
		# c) C parts
		ind_C = 2*N + ii*nr;
		NX[ind_C:ind_C+nr,0] = J_PSI_C_hat[:,ii]#IR.dot(J_PSI_C_hat[:,ii]) # Multip 1/r^2

	return -NX; # DF_X(X')

#@profile
def LIN_LX(gr,T0_JT,A4,NABLA2,Tau,S,X_in):

	N = A4.shape[0]
	s = (3*N,1);

	if X_in.shape != s:
		#print "Cast X_in to shape ",X_in.shape
		X = np.zeros((3*N,1)); X[:,0] = X_in;
		#print "Casted X to shape ",X.shape
	else:
		X = X_in		
	T0_PSI = np.zeros((N,1)); RAG = np.zeros((N,1)); 
	LX = np.zeros((3*N,1));

	# Linear
	RAG[:,0] = gr.dot(X[N:2*N,0] - S*X[2*N:3*N,0]); # S here is a parameter
	T0_PSI[:,0] = T0_JT.dot(X[0:N,0]); # Add T'0*psi

	# 1) Vorticity
	#NX[0:N] += RAG + A4.dot(X[0:N])
	LX[0:N] = RAG + A4.dot(X[0:N])# <12%

	# 2) Temperature
	#NX[N:2*N] = T0_PSI + NABLA2.dot(X[N:2*N]);
	LX[N:2*N] = T0_PSI + NABLA2.dot(X[N:2*N]); # <12 ?%

	# 3) Concentration
	#NX[2*N:3*N] += T0_PSI +  Tau*NABLA2.dot(X[2*N:3*N]);
	LX[2*N:3*N] = T0_PSI +  Tau*NABLA2.dot(X[2*N:3*N]); # 12%}

	return LX;

#@profile
def PRECOND(LINV_psi_SP, LINV_T_SP, LINV_C_SP, NX):

	N = LINV_psi_SP.shape[0]
	# 1) Vorticity
	NX[0:N] = LINV_psi_SP.dot(NX[0:N]) # <12%

	# 2) Temperature
	#NX[N:2*N] += T0_PSI + NABLA2.dot(X[N:2*N]);
	NX[N:2*N] = LINV_T_SP.dot(NX[N:2*N]); # <12 ?%

	# 3) Concentration
	#NX[2*N:3*N] += T0_PSI +  Tau*NABLA2.dot(X[2*N:3*N]);
	NX[2*N:3*N] = LINV_C_SP.dot(NX[2*N:3*N]); # 12%	

	return NX;


def A4_BSub(A,g):

	N = nr*N_fm; f = np.zeros((N,1)) 

	for jj in xrange(N_fm):		
		ind_j = (N_fm - (jj + 1) )*nr; # Row ind

		b = np.zeros((nr,1))
		
		# Only want to sample every second one as even
		if jj%2 == 0:
			# Evens
			ITER = np.arange(0,jj,2)
		elif jj%2 == 1:
			# odds
			ITER = np.arange(1,jj,2)

		for kk in ITER: # odds
			ind_k = ( N_fm - (kk + 1) )*nr # Column inds
			#b += np.dot(A[ind_j:ind_j+nr,ind_k:ind_k+nr],f[ind_k:ind_k+nr])

			# Pre-invert this line ?
		f[ind_j:ind_j+nr] = np.linalg.solve(A[ind_j:ind_j+nr,ind_j:ind_j+nr],g[ind_j:ind_j+nr]-b)

	return f;

def MULT(A,g,f):

	N = nr*N_fm; #f = np.zeros((N,1)) 

	for jj in xrange(N_fm):		
		ind_j = (N_fm - (jj + 1) )*nr; # Row ind
				
		# Only want to sample every second one as even
		if jj%2 == 0:
			# Evens
			ITER = np.arange(0,jj+1,2)
		elif jj%2 == 1:
			# odds
			ITER = np.arange(1,jj+1,2)
			
		for kk in ITER:#xrange(jj+1):#ITER: # odds
			ind_k = (N_fm - (kk + 1) )*nr # Column inds
			
			f[ind_j:ind_j+nr] += A[ind_j:ind_j+nr,ind_k:ind_k+nr].dot(g[ind_k:ind_k+nr]);

	return f;	

def R2(R,N_fm): # Correct
	GR = diags( (R[1:-1]**2),0,format="csr"); 
	#GR = np.diag( (R[1:-1]**2) ) 
	
	AT = [];
	for jj in xrange(N_fm): 
		AT.append(GR);
	
	return block_diag(AT,format="csr");#.todense();

def T0(R,N_fm,d): # Correct
	#const  = 1.0/dct(np.cos(0.*x),type=2,norm='ortho')[0];
	#print const
	A_t = -(1.0+d)/d; IR = -A_t*diags( 1.0/(R[1:-1]**2),0,format="csr"); 
	
	AT = [];
	for jj in xrange(N_fm): 
		AT.append(IR);
	
	return block_diag(AT,format="csr");#.todense();	

#*~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~~~~~~~~~~ *~~~~~~~~~~

def EigenVals(A):

	EIGS = np.linalg.eigvals(A); # Modify this to reduce the calculation effort ??
	idx = EIGS.argsort()[::-1];
	eigenValues = EIGS[idx];
	print "EigenVals ", eigenValues[0:10], "\n" ## Comment Out if possible
	
	return eigenValues[0:5];

def EigenVECS(A,R):

	eigenValues, eigenVectors = np.linalg.eig(A); # Modify this to reduce the calculation effort ??
	
	# Sort eigenvalues
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	print "EigenVals ",eigenValues[0:10]
	
	'''
	plt.title(r'$\lambda$ Eigenvalues')
	plt.plot(eigenValues.imag,eigenValues.real,'bo',markerfacecolor='none') #marker='o',markerfacecolor='none', markeredgecolor='blue')
	#plt.plot(np.linspace(0.5,-0.5,10),np.zeros(10),'k-',linewidth=1.0)
	plt.xlabel(r'$\Im \{ \lambda \}$',fontsize=20)
	plt.ylabel(r'$\Re \{ \lambda \}$',fontsize=20)
	#plt.ylim([-100.0,20.0])
	#plt.xlim([20.0,-20.0])
	plt.show()


	LEN = eigenVectors.shape[0]; 
	nr = len(R[1:-1])
	#sys.exit()
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	plt.title(r'Eigenvectors')
	plt.xlabel(r'radial - $r$',fontsize=20)
	plt.ylabel(r'function $\psi,T,C$',fontsize=20)
	#plt.ylim([-100.0,20.0])
	#plt.xlim([20.0,-20.0])

	xx = np.linspace(1.0,1.0+d,50)
	x = np.zeros((nr+2,1))
	print x.shape
	for ii in xrange( 3 ):
		
		print "k = ",ii+1
		ind = ii*nr
		x = np.zeros((nr+2,1))
		x[1:-1] = eigenVectors[ind:ind+nr,0].real;
		p  = np.linalg.norm(x,2);
		if p > 1e-13:
			x = x/p;
			plt.plot(xx,np.polyval(np.polyfit(R,x,nr+2),xx),'-',label =r'$\psi_k=%s$'%ii) #marker='o',markerfacecolor='none', markeredgecolor='blue')
	
		ind1 = N_fm*nr + ii*nr
		x = np.zeros((nr+2,1))
		x[1:-1] = eigenVectors[ind1:ind1+nr,0].real;
		p  = np.linalg.norm(x,2);
		if p > 1e-13:
			x = x/p;
			plt.plot(xx,np.polyval(np.polyfit(R,x,nr+2),xx),'--',label =r'$T_k=%s$'%ii)

		ind2 = 2*N_fm*nr + ii*nr
		x = np.zeros((nr+2,1))
		x[1:-1] = eigenVectors[ind2:ind2+nr,0].real;
		p  = np.linalg.norm(x,2);
		if p > 1e-13:
			x = x/p;
			plt.plot(xx,np.polyval(np.polyfit(R,x,nr+2),xx),':',label =r'$C_k=%s$'%ii)
	
	plt.grid()
	plt.legend()
	plt.show()
	'''
	return eigenVectors[:,0]

# Caution in switching from sparse to dense!!!!
#M = M_0(D,R,N_fm).todense();
#L = L_0(D,R,N_fm,d,Pr,Ra,Tau).todense()
#A = np.matmul(inv(M).todense(),L.todense()) # Preform as Sparse
#X = EigenVECS(A,R)
#sys.exit()