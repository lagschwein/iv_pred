import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 

def load_model(dd = './reg_models/', model='ridge', lag=2):
    return pickle.load(open(dd + 'model_%s_ts_%d_figs.pkl' % (model, lag), 'rb'))

def plot_feature_importance(coeff, lag=2):
    # calculate the average absolute coefficient value for each feature
    avg_abs_coeff = np.mean(np.abs(coeff), axis=0)

    # normalize coefficients
    norm_coeff = avg_abs_coeff/ np.sum(avg_abs_coeff)
    print(norm_coeff.shape)

    # Plot bar plot
    fig, ax = plt.subplots(layout='constrained', figsize=(15, 6))
    plt.bar(range(len(norm_coeff)), norm_coeff)
    plt.xlabel('Feature Index', labelpad=30)
    plt.ylabel('Normalized Average Absolute Coefficient')
    plt.title('Feature Importance in multi-output %s model with lag %d' % ('Ridge', lag))
    plt.xticks(range(0, len(norm_coeff), 5), range(0, len(norm_coeff), 5))
    # Plot extra tick to show days which correspond to np.arange(coeff.shape[0], coeff.shape[1], coeff.shape[0])
    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(np.arange(coeff.shape[0], coeff.shape[1]+1, coeff.shape[0]), ['\nDay ' + str(i) for i in range(1, (coeff.shape[1]//coeff.shape[0])+1, 1)])
    plt.show()
    # plt.figure(figsize=(10, 6))
    # for i in range(coeff.shape[0]):
    #     plt.plot(coeff[i], label='Feature Importance')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Feature Importance')
    # plt.title('Feature Importance')
    # # Split the ticks into coeff.shape[1]/lag intervals labeled day 1 to day n
    # plt.xticks(np.arange(coeff.shape[0], coeff.shape[1], coeff.shape[0]), [str(i) for i in range(1, coeff.shape[1]//coeff.shape[0], 1)])
    # plt.tight_layout()
    # plt.grid(True)
    # plt.set_cmap('winter')
    # plt.show()


if __name__ == '__main__':
    lag = 10
    model = load_model(model='enet', lag=lag)
    plot_feature_importance(model.coef_, lag)