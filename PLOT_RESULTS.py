#!/usr/bin/env python

import numpy as np

from scipy.fftpack import dct, idct, dst, idst

import sys, os, time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

#----------- Latex font ----
import matplotlib
matplotlib.use('TkAgg') # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation

rc('font',**{'family':'serif','serif':['Palatino'],'weight':'bold','size':20})
rc('text', usetex=True)
#------------------------------

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

#Stringy = ['Results/Test_l21_02P'];
Stringy = ['Test_l21_13P'];

from os.path import join
'''
image_dir = '../input/dog-breed-identification/train/'
img_paths = [join(image_dir, filename) for filename in 
                           ['0c8fe33bd89646b678f6b2891df8a1c6.jpg',
                            '0c3b282ecbed1ca9eb17de4cb1b6e326.jpg',
                            '04fb4d719e9fe2b6ffe32d9ae7be8a22.jpg',
                            '0e79be614f12deb4f7cae18614b7391b.jpg']]
'''
# 2) Make Directory & Save Files, USE PANDAS

#Stringy = ['Test_l20_10P'];

# 3) Change into Results folder
os.chdir("".join(Stringy))	

Parameters = np.load("Parameters.npy");

print("#~~~~~~~~~~~~~~~~ Control Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
Ra_c = Parameters[0]
eps = Parameters[1]
Tau = Parameters[2]
Pr = Parameters[3]
d = Parameters[4]
Ra_s = Parameters[5]

print(" #~~~~~~~~~~~~~~~~ Numerical Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
dt = Parameters[6];
Nr = int(Parameters[7]);
N_fm = int(Parameters[8]);
nsave = Parameters[9];

D,R = cheb_radial(Nr,d); nr = len(R[1:-1])

X_Norm  = np.load("X_Norm_L2.npy")
Time = np.load("Time.npy")
X_DATA = np.load('X_VID_l20.npy')


plt.figure(figsize=(10,8))
plt.plot(Time,X_Norm,'k-',linewidth=1.1)
plt.xlabel(r'time $t$',fontsize=25)
plt.ylabel(r'$||X||$',fontsize=25)
#plt.xlim([np.min(Time),np.max(Time)])
#plt.xlim([9.9,10])
plt.savefig("X_vs_TIME_FULL.eps",format='eps', dpi=1800)
plt.show()

# Plot on a spherical shell
#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~
X = X_DATA[-1]
from Plot_Tools import * 
#N_fm = 500
Theta_grid = np.linspace(0,np.pi,N_fm); 
r_grid = np.linspace(R[-1],R[0],50);
PSI, T, C,T_0 = SPEC_TO_GRID(R,r_grid,N_fm,X,d)
#Plot_Package_CHE(r_grid,Theta_grid,C+T_0,PSI,T+T_0,d)

Energy(X,N_fm,R);

# A) Plot on a plane layer for larger
#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~



# 1) Fix \theta labels to be [0,pi]
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,figsize=(24,12),dpi=200)
RES = 15; # Set Contour Levels

C_cntr = ax1.contour(Theta_grid,r_grid,T+T_0, colors = 'k',levels=RES, linewidths=0.5);
C_cntrf = ax1.contourf(Theta_grid,r_grid,T+T_0, levels=RES, cmap="RdBu_r")
fig.colorbar(C_cntrf, ax=ax1)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.set_title(r'$T$',fontsize=20)


P_cntr = ax2.contour(Theta_grid,r_grid,PSI, colors = 'k',levels=RES, linewidths=0.5);
P_cntrf = ax2.contourf(Theta_grid,r_grid,PSI, levels=RES, cmap="RdBu_r")
fig.colorbar(P_cntrf, ax=ax2)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax2.set_title(r'$\psi$',fontsize=20)


T_cntr =ax3.contour(Theta_grid,r_grid,C+T_0, colors = 'k',levels=RES, linewidths=0.5);
T_cntrf = ax3.contourf(Theta_grid,r_grid,C+T_0, levels=RES, cmap="RdBu_r")
fig.colorbar(T_cntrf, ax=ax3)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax3.set_ylabel(r'$r$',fontsize=20)

ax3.set_xlabel(r'$\theta$',fontsize=25)
ax3.set_title(r'$C$',fontsize=25)


#plt.ylabel(r'r',fontsize=20)
#plt.xlabel(r'$\theta$',fontsize=20)
plt.subplots_adjust(hspace=0.25)
plt.savefig("X_Frame.eps",format='eps', dpi=1800)
plt.show()

#~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~~~~~~~~~~~~~~~~#~~~~~~~~~~~

# B) Make a video

RES = 15; # Set Contour Levels

def animate(k):
	SKIP = 1; 
	X = X_DATA[500+SKIP*k];
	PSI, T, C,T_0 = SPEC_TO_GRID(R,r_grid,N_fm,X,d)
	C = C + T_0;
	T = T + T_0;

	print "Frame k ",k,"\n"
	global PSI_cont, PSI_contf, T_cont, T_contf;
	global C_cont, C_contf
	global TEXT
	
	try:
		ax1.collections = [] 	
		C_contf = ax1.contourf(Theta_grid,r_grid,T, levels=RES, cmap="RdBu_r")
		C_cont = ax1.contour(Theta_grid,r_grid,T, colors = 'k',levels=RES, linewidths=0.5)
		
	except ValueError:  #raised if `y` is empty.
		pass

	try:
		ax2.collections = [] 
		PSI_contf = ax2.contourf(Theta_grid,r_grid,PSI, levels=RES, cmap="RdBu_r")
		PSI_cont = ax2.contour(Theta_grid,r_grid,PSI, colors = 'k',levels=RES, linewidths=0.5)
		
	except ValueError:  #raised if `y` is empty.
		pass	

	
	try:
		ax3.collections = [] 	
		T_contf = ax3.contourf(Theta_grid,r_grid,C, levels=RES, cmap="RdBu_r")
		T_cont = ax3.contour(Theta_grid,r_grid,C, colors = 'k',levels=RES, linewidths=0.5)
		
	except ValueError:  #raised if `y` is empty.
		pass

	# Inlcude Frame time	
	print 'Frame Number: ', SKIP*k
	Time = nsave*float(SKIP*k)*dt     		
	st = r'$Time \quad (\alpha/r^2_1)t = \quad %4.4f $' % Time  #(0.25*float(k))#(k*dt)/w_ref
	#time_text = plt.text(0.15, 0.1, st , color='black', fontsize = 40)
	time_text.set_text(st)
	'''plt.pause(0.005);
	plt.draw();
	time.sleep(0.01)
	fig.clf();
	'''
	return T_cont, T_contf, PSI_cont, PSI_contf, C_cont, C_contf


# D) Plot a video, Uses function defined above 
# Video
import matplotlib.animation as animation

plt.ion();
plt.rc('axes', linewidth=2)
tsteps=100; ### NUMBER OF FRAMES THAT WILL BE CREATED

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,figsize=(24,12),dpi=200)
RES = 25; # Set Contour Levels

C_cntr = ax1.contour(Theta_grid,r_grid,T+T_0, colors = 'k',levels=RES, linewidths=0.5);
C_cntrf = ax1.contourf(Theta_grid,r_grid,T+T_0, levels=RES, cmap="RdBu_r")
fig.colorbar(C_cntrf, ax=ax1)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax1.set_title(r'$T$',fontsize=25)


PSI_cntr = ax2.contour(Theta_grid,r_grid,PSI, colors = 'k',levels=RES, linewidths=0.5);
PSI_cntrf = ax2.contourf(Theta_grid,r_grid,PSI, levels=RES, cmap="RdBu_r")
fig.colorbar(P_cntrf, ax=ax2)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax2.set_title(r'$\psi$',fontsize=25)


T_cntr =ax3.contour(Theta_grid,r_grid,C+T_0, colors = 'k',levels=RES, linewidths=0.5);
T_cntrf = ax3.contourf(Theta_grid,r_grid,C+T_0, levels=RES, cmap="RdBu_r")
fig.colorbar(T_cntrf, ax=ax3)#
#ax1.set(xlim=(-2, 2), ylim=(-2, 2))
ax3.set_title(r'$C$',fontsize=25)

ax3.set_ylabel(r'$r$',fontsize=25)
ax3.set_xlabel(r'$\theta$',fontsize=25)

plt.subplots_adjust(hspace=0.25)
#plt.show()


k=1
#sShould coresspond to Fac
Time = nsave*float(k)*dt     		
st = r'$Time \quad (\alpha/r^2_1)t = \quad %4.4f $' % Time  #(0.25*float(k))#(k*dt)/w_ref
#TEXT = plt.figtext(0.15, 0.1, st , color='black', fontsize = 20 ) #, weight='roman') #,size='x-small')

time_text = plt.figtext(0.05,0.05, st , color='black', fontsize = 30)

#plt.show()
# Get Start Time
start_time = time.time()

'''
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')

# Change the video bitrate as you like and add some metadata.
writer = FFMpegWriter(fps=15, bitrate=1000, metadata=metadata)
'''

anim = animation.FuncAnimation(fig, animate, frames=tsteps,repeat=False, interval=30)
#anim.save('animation1.mp4',fps=2,codec="libx264",extra_args=['-pix_fmt', 'yuv420p'])
#anim.save('Animation_24.mp4',fps=3,codec="libx264") #,extra_args=['-pix_fmt', 'yuv420p'],dpi=200)
anim.save('SIM.mp4',fps=1,codec='h264',bitrate=1000);#,dpi=500)
#anim.save('Chaotic_Plume.mp4',fps=4,codec="libx264",bitrate=1000);#,dpi=500)
#writer=animation.FFMpegWriter(bitrate=500)
#anim.save('Animation_13.mp4',writer=writer,fps=8)

# Print Out time taken
end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))	