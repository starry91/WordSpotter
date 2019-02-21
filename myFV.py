import glob
import argparse
import cv2
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA

PCA_obj = None

def getSIFTfeatures(path, N = 50):
    try:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (256, 256))
        _, descriptors = cv2.xfeatures2d.SIFT_create(
            nfeatures=N).detectAndCompute(img, None)
        if(descriptors.shape[1]*descriptors.shape[0] != 50*128):
            return None
        return descriptors
    except:
        print(file)   

def getDirFeatures(dir_path):
    return np.array([getSIFTfeatures(filepath) for filepath in dir_path])

def applyPCA(train_data):
    if(PCA_obj is None):
        PCA_obj = PCA()
        PCA_obj.fit(train_data)
    res = PCA_obj.transform(train_data)
    return res

def generate_gmm(features, N):
    model = mixture.GaussianMixture(N)
    model.fit(features)
    g_mean, g_covar, g_weights = model.means_, model.covariances_, model.weights_
    np.save("means.gmm", g_mean)
    np.save("covs.gmm", g_covar)
    np.save("weights.gmm", g_weights)
    return g_mean, g_covar, g_weights

def getAllFeatures(working_folder):
    dirs = glob.glob(working_folder + "/*")
    SIFT_features = np.array([getDirFeatures(folder) for folder in dirs])
    reduced_SIFT_features = applyPCA(SIFT_features)   
    return reduced_SIFT_features


def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(
        x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def computeStatistics(SIFT_features, g_means, g_covars, g_weights):
    gaussians, s0, s1, s2 = {}, {}, {}, {}
    for i in range(len(g_weights)):
        s0[i] = 0, s1[i] = 0, s2[i] = 0

    g = [multivariate_normal(mean=g_means[k], cov=g_covars[k], allow_singular=False) for k in range(len(weights))]
    for index, x in enumerate(SIFT_features):
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
    # print("gaussians: {0}".format(gaussians))

    for k in range(0, len(weights)):
        for index, x in enumerate(SIFT_features):
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
    return s0, s1, s2

# def fisherSignature(stats, )

def fisher4image(path, gmm):
    g_means, g_covars, g_weights = *gmm
    features = getSIFTfeatures(path)
    s0, s1, s2 = computeStatistics(features)
    g_alpha = []
    g_mu = []
    g_var = []
    T = len(features)
    for k in range(len(s0)):
        g_alpha[k] = (s0[k] - T*g_weights[k])/np.sqrt(g_weights[k])
        g_mu = (s1[k] - g_means[k]*s0[k])/(np.sqrt(g_weights[k]*g_covars[k])
        g_var = (s2[k] -2*g_means[k]*s1[k] + (np.square(g_means[k]) - g_var)*s0[k])/(np.sqrt(2*g_weights[k])*g_covars[k])
    pass



def fisherFromFolder(folder, gmm)
    [fisher4image(path) for path in glob.glob(folder + "/*.JPG")]


def getFisherFeatures(folder_path, gmm):
    [fisherFromFolder(folder, gmm) for folder in glob.glob(folder_path + "/*")]
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', "--dir", help="Directory with images", default='.')
    parser.add_argument("-g", "--loadgmm", help="Load Gmm dictionary",
                        action='store_true', default=False)
    parser.add_argument(
        '-n', "--number", help="Number of words in dictionary", default=1, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    working_folder = args.dir
    print(working_folder)
    print("no. of weights {0}".format(args.number))
    train_features = getAllFeatures(working_folder)
    #load_gmm(working_folder, train_features) if args.loadgmm else 
    gmm = generate_gmm(train_features, args.number)
    fisher_features = getFisherFeatures(working_folder, gmm)
    # TBD, split the features into training and validation
    classifier = train(gmm, fisher_features)
    rate = success_rate(classifier, fisher_features)
    print("Success rate is", rate)
g_means, g_covars, g_weights