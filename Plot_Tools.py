#! /usr/bin/env python

#----------- Latex font ----
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino'],'weight':'bold','size':20})
rc('text', usetex=True)
#------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, dst, idst
import math
RES = 25;

def Energy(X,N_Modes,R):
	
	plt.figure(figsize=(8, 6))

	nr = len(R[1:-1]); N = N_Modes*nr;

	PSI = X[0:N]; T = X[N:2*N]; C = X[2*N:3*N];

	E_PSI = np.zeros(N_Modes);
	E_T = np.zeros(N_Modes);
	E_C = np.zeros(N_Modes);

	for k in xrange(N_Modes):
		ind = k*nr;
		E_PSI[k] = np.linalg.norm(PSI[ind:ind+nr,0],2);
		E_T[k] = np.linalg.norm(T[ind:ind+nr,0],2);
		E_C[k] = np.linalg.norm(C[ind:ind+nr,0],2);	
	#E = np.log10(E[:]);
	#print E
	#E[np.isneginf(E)] = 0;
	#print E
	LLc = np.arange(0,N_Modes,1)[:]
	LLs = np.arange(1,N_Modes+1,1)[:]
	#E = [];
	#for l in xrange(N_Modes):
	#	plt.semilogy(l,E[l], 'b:')

	plt.semilogy(LLs[:],E_PSI[:], 'k-',label=r'$\psi$')
	plt.semilogy(LLc[:],E_T[:], 'r-',label=r'$T$')
	plt.semilogy(LLc[:],E_C[:], 'b-',label=r'$C$')

	plt.xlabel(r'Mode number $k$', fontsize=16)
	plt.ylabel(r'$log10( ||X_{\ell}|| )$', fontsize=16)
	#plt.xticks(np.arange(0,N_Modes,10))
	plt.xlim([0,N_Modes])
	plt.grid()
	plt.legend()
	plt.savefig("X_SPECTRUM.eps",format='eps', dpi=1800)
	plt.show()

def SPEC_TO_GRID(R,xx,N_fm,X,d): # Fix this code to extend to the boundaries

	
	# 2) Normalization Weights, must be inverses
	x = np.linspace(0,np.pi,N_fm);
	const = 1.0/dct(np.cos(0.0*x),type=2,norm='ortho')[0];
	
	#print const
		
	Nr = len(R); t_0 = np.zeros(len(R)); 
	# Base State
	A_T, B_T = -(1.0+d)/d, -1.0/d; alpha = 1.0+d;
	for ii in xrange(Nr):
		t_0[ii] = (-A_T/R[ii] + B_T)
	TR_0 = np.polyval(np.polyfit(R,t_0,Nr),xx)
	T_0 = np.outer(TR_0,np.ones(N_fm));
	

	# A) Computational grid
	nr = len(R[1:-1])
	s = (nr,N_fm); N = nr*N_fm
	Vort = np.zeros(s); Temp = np.zeros(s); Conc = np.zeros(s)
	
	# 1) Transform vector to sq grid
	for ii in xrange(N_fm):	
		ind = ii*nr
		Vort[:,ii] = X[ind:ind+nr,0].reshape(nr);
			
		ind = N + ii*nr
		Temp[:,ii] = X[ind:ind+nr,0].reshape(nr);

		ind = 2*N + ii*nr
		Conc[:,ii] = X[ind:ind+nr,0].reshape(nr);
		#if ii == 0:
		#	Temp[:,ii] = Temp[:,ii] + t_0[1:-1];
		#	Conc[:,ii] = Conc[:,ii] + t_0[1:-1]

	# 2) Take the idct, idst of each radial level set
	for jj in xrange(nr):
		Vort[jj,:] = idst(Vort[jj,:],type=2,norm='ortho')
		Temp[jj,:] = idct(Temp[jj,:],type=2,norm='ortho') # Assuming these must be normalized?
		Conc[jj,:] = idct(Conc[jj,:],type=2,norm='ortho')


	# B) Visualisation grid
	s = (len(xx),N_fm);
	PSI, T, C = np.zeros(s), np.zeros(s), np.zeros(s); 	
	
	
	# 3) Polyfti
	for ii in xrange(N_fm):
		
		ind = ii*nr
		
		psi = np.hstack((0,Vort[:,ii],0));
		PSI[:,ii] = const*np.polyval(np.polyfit(R,psi,Nr),xx)
		#psi = Vort[:,ii]
		#PSI[:,ii] = np.polyval(np.polyfit(R[1:-1],psi,nr),xx)
		
		t = np.hstack((0,Temp[:,ii],0))
		T[:,ii] = const*np.polyval(np.polyfit(R,t,Nr),xx)
		#t = Temp[:,ii]
		#T[:,ii] = const*np.polyval(np.polyfit(R[1:-1],t,nr),xx)
		
		c = np.hstack((0,Conc[:,ii],0))
		C[:,ii] = const*np.polyval(np.polyfit(R,c,Nr),xx)
		#c = Conc[:,ii]
		#C[:,ii] = const*np.polyval(np.polyfit(R[1:-1],c,nr),xx)	
		

	return PSI, T, C,T_0;
	#return Vort, Temp, Conc,T_0;

def Plot_Package_CHE(R,theta,omega,psi,thermal,sigma): # Returns Array accepted by contourf - function

	#-- Make Really Narrow Slice to form Axis ---------------
	NN = 20
	azimuths = np.linspace(0,1e-08, NN)
	zeniths = np.linspace(0,5, NN )
	s = (NN,NN)
	ww = np.zeros(s) 

	alpha = 1.0+sigma;

	'''#---- Repackage AA into omega[i,j] -------------
	nr, nth = len(R), len(theta)
	s = (nr,nth)
	omega, psi, thermal = np.zeros(s), np.zeros(s),np.zeros(s)
	row, col = 0,0
	for i in range(len(R)):
		col = i;
		for j in range(len(theta)): # Very Bottom and top rows must remain Zero therefore 
			omega[i,j] = OMEGA[col,t];
			psi[i,j] = PSI[col,t];
			thermal[i,j] = THERMAL[col,t];
			col = col + nr; 
	'''
	#if plot_out == True:
	fig, ax = plt.subplots(1,3,subplot_kw=dict(projection='polar'),figsize=(16,6))  
	###fig.suptitle(r'Reynolds Number $Re_1 = %.1f$, Rayleigh Number $Ra = %.1f$, Separation $d = %s$'%(Re1,Ra,sigma), fontsize=16)      
	
	# --------------- Plot Omega -----------
	ax[0].contourf(azimuths,zeniths,ww)
	try:
		#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
		p1 = ax[0].contourf(theta,R,omega,RES) 
		ax[0].contour(theta,R,omega,RES)
		ax[0].contourf(2.0*np.pi-theta,R,omega,RES) 
		ax[0].contour(2.0*np.pi-theta,R,omega,RES)
		#p1 = ax[0].contourf(theta,R,omega,RES) 
		#ax[0].contour(theta,R,omega,RES)#	, colors = 'k',linewidths=0.7) #,RES)	
		#ax[0].clabel(p1, fmt='%2.1f', colors='w', fontsize=14)	
	except ValueError:
		pass
	ax[0].set_theta_zero_location("S")
	ax[0].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[0].set_ylim(0,alpha)
	ax[0].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#print omega.ax(axis=1).max()
	#print omega.min(axis=1).min()
	ax[0].set_xlabel(r'$C_{max,min} = (%.3f,%.3f)$'%(omega.max(axis=1).max(),omega.min(axis=1).min()), fontsize=20) #, color='gray')

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[0].set_title(r'$C$', fontsize=16, va='bottom')
	cbaxes = fig.add_axes([0.05, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar1 = plt.colorbar(p1, cax = cbaxes)

	# ---------------- PSI Stream Function --------------------       
	ax[1].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		p2 = ax[1].contourf(theta,R,psi,RES) 
		ax[1].contour(theta,R,psi,RES)
		ax[1].contourf(2.0*np.pi-theta,R,psi,RES) 
		ax[1].contour(2.0*np.pi-theta,R,psi,RES) #-TT	
		#p2 = ax[1].contourf(theta,R,psi,RES) 
		#ax[1].contour(theta,R,psi,RES)#	, colors = 'k',linewidths=0.7) #
	except ValueError:
		pass	
	ax[1].set_theta_zero_location("S")
	ax[1].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[1].set_ylim(0,alpha)
	ax[1].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	ax[1].set_xlabel(r'$\Psi_{max,min} = (%.3f,%.3f)$'%(psi.max(axis=1).max(),psi.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[1].set_title(r'$\Psi$', fontsize=16, va='bottom')
	#cbaxes1 = fig.add_axes([0.6, 0.25, 0.015, 0.4]) # left, bottom, height, width
	#cbar2 = plt.colorbar(p2, cax = cbaxes1)

	# ----------------- Temperature Field -----------------
	ax[2].contourf(azimuths,zeniths,ww)

	#matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
	try:
		## TO modify for low value contours
		p3 = ax[2].contourf(theta,R,thermal,RES) #-TT
		ax[2].contour(theta,R,thermal,RES) #-TT	
		ax[2].contourf(2.0*np.pi-theta,R,thermal,RES) #-TT	
		ax[2].contour(2.0*np.pi-theta,R,thermal,RES) #-TT	
		#p3 = ax[2].contourf(theta,R,thermal,RES) 
		#ax[2].contour(theta,R,thermal,RES)#	, colors = 'k',linewidths=0.7) #,RES )
		#ax[2].clabel(CS, inline=1, fontsize=10)		
	except ValueError:
		pass
	ax[2].set_theta_zero_location("S")
	ax[2].bar(math.pi, 0.0 )

	# Adjust the axis
	ax[2].set_ylim(0,alpha)
	ax[2].set_rgrids([0.5,1,alpha], angle=345.,fontsize=12)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	# ax.set_ylabel(r'\textbf{Radial position} (R)')
	#ax[2].set_xlabel(r'\textit{Polar Angel} ($\theta$) \quad\quad \textit{Radial position} (r)', fontsize=12) #, color='gray')
	ax[2].set_xlabel(r'$T_{max,min} = (%.3f,%.3f)$'%(thermal.max(axis=1).max(),thermal.min(axis=1).min()), fontsize=20)

	# Make space for title to clear
	plt.subplots_adjust(top=0.8)
	ax[2].set_title(r'$T$', fontsize=16, va='bottom')
	cbaxes2 = fig.add_axes([0.95, 0.25, 0.015, 0.4]) # left, bottom, height, width
	cbar3 = plt.colorbar(p3, cax=cbaxes2)
	
	#branch = 'RL'; # 'SL' 'RL'
	#STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	'''
	if Pr >= 1.0:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr',str(int(Pr)),'.eps'])
	elif Pr == 0.1:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr01.eps']) 
	elif Pr == 0.01:
		STR = "".join([branch,'_Solution_Re',str(int(Re1)),'_Pr001.eps']) 
	'''	
	#plt.savefig(STR, format='eps', dpi=1800)
	plt.show()
				
	#return omega, ksi, psi	