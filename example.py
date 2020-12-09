import numpy as np
import scipy.io as sio
import counts_analysis as ca
import matplotlib.pyplot as plt

dat = sio.loadmat('sample_dat.mat') # dat['spike_counts'] is matrix that is # of trials x # of neurons

sc_obj = ca.counts_analysis(dat['spike_counts'],dat['targ_angs'].flatten(),dat['bin_size'])

# check that average firing rate is properly computed
plt.figure(0)
chan_idx = np.arange(0,dat['spike_counts'].shape[1])+1
avg_fr = sc_obj.compute_avg_fr()
plt.bar(chan_idx,avg_fr)
plt.xlabel('channel index')
plt.ylabel('average rate (spikes/s)')
plt.show()

# check that condition means and tuning curves are properly computed
cond_means = sc_obj.compute_cond_means()
tune_params,tune_pred,ang_pred = sc_obj.fit_cosine_tuning(compute_p=True,rand_seed=0)
plt.figure(1)
for i in range(dat['spike_counts'].shape[1]):
    plt.plot(sc_obj.lbls,cond_means[:,i],marker='o',ls='None')
    plt.plot(ang_pred,tune_pred[:,i],marker='None',ls='-',color='r')
    plt.xticks(ticks=sc_obj.lbls)
    plt.xlabel('target angle')
    plt.ylabel('average count (spikes)')
    plt.title('p = {:.3f}'.format(tune_params['p_val'][i]))
    plt.show()
    tmp = input('Press ENTER to continue or c for next neuron...')
    if tmp.lower()=='c':
        plt.clf()
    else:
        break

# check that condition means are properly removed
X_nomean = sc_obj.rm_cond_means()
cond_var = sc_obj.get_cond_varexp(return_each=True)
print('On average, conditions explain {:.2f}% var'.format(sc_obj.get_cond_varexp()))
print('   Max: {:.2f}%\n   Min: {:.2f}%'.format(np.max(cond_var),np.min(cond_var)))
plt.figure(2)
plt.plot(sc_obj.X[:,0])
plt.plot(X_nomean[:,0] + avg_fr[0])
plt.xlabel('trial index')
plt.ylabel('neuron 1, counts')
plt.legend(('original','no cond mean'))
plt.title('conditions explain {:.2f}% var'.format(cond_var[0]))
plt.show()

# check that fano factor is properly computed
fano,sc_mean,sc_var = sc_obj.compute_fano(return_stats=True)
plt.figure(3)
plt.plot(sc_mean,sc_var,'bo')
x=np.array([0,np.max(np.concatenate((sc_mean,sc_var)))])
plt.plot(x,x,'k--')
plt.xlabel('mean spike count')
plt.ylabel('var spike count')
plt.show()

# check that autoregression is properly computed
no_ar,ar_proc = sc_obj.rm_autoreg(order=25,both_dirs=True,auto_type='mean',fa_remove=True,fa_dims=10)
ar_var = sc_obj.get_auto_varexp(return_each=True,order=25,both_dirs=True,auto_type='mean')
plt.figure(4)
for i in range(dat['spike_counts'].shape[1]):
    plt.subplot(1,2,1)
    plt.plot(X_nomean[:,i],linewidth=.5)
    plt.plot(ar_proc[:,i],linewidth=.5)
    ylims = plt.ylim()
    plt.title('arT explains {:.2f}% var'.format(ar_var[i]))
    plt.subplot(1,2,2)
    plt.plot(no_ar[:,i],linewidth=.5)
    plt.ylim(ylims)
    plt.show()
    tmp = input('Press ENTER to continue or c for next neuron...')
    if tmp.lower()=='c':
        plt.clf()
    else:
        break
        
# check decoding accuracy (with and without removing autoreg)
acc,_ = sc_obj.decode(rm_auto=False,rand_seed=0)
acc_rmauto,_ = sc_obj.decode(rm_auto=True,rand_seed=0,auto_order=25,auto_type='mean')
chance_lvl = 1/sc_obj.n_lbls * 100
print('Decoding:\n   chance {:.2f}%\n   original {:.2f}%\n   no auto {:.2f}%\n'.format(chance_lvl,acc*100,acc_rmauto*100))

# check rsc and signal correlation computation
rsc,rsc_null = sc_obj.compute_rsc(compute_null=True,rand_seed=0)
sig_corr = sc_obj.compute_signal_corr()
plt.figure(5)
plt.subplot(1,2,1)
plt.hist(sig_corr,bins=np.arange(-1,1.01,.05),histtype='step')
plt.xlabel('signal corr')
plt.subplot(1,2,2)
plt.hist(rsc,bins=np.arange(-.5,.51,.05),histtype='step')
plt.hist(rsc_null,bins=np.arange(-.5,.51,.05),histtype='step')
plt.xlabel('rsc')
plt.legend(('X','null'))
plt.show()

null_cuts = np.percentile(rsc_null,[2.5,97.5])
rsc_low,rsc_high = rsc<null_cuts[0], rsc>null_cuts[1]
sig_lowrsc = np.mean(sig_corr[rsc_low])
sig_highrsc = np.mean(sig_corr[rsc_high])
print('average signal correlation:\n   neg rsc (p<.05): {:.3f}\n   pos rsc (p<.05): {:.3f}'.format(sig_lowrsc,sig_highrsc))
