# Author: Jacob Gildenblat, 2014
# License: you may use this for whatever you like
import sys
import glob
import argparse
import numpy as np
import math
import cv2
from scipy.stats import multivariate_normal
from sklearn import mixture
import time
from sklearn import svm
from sklearn.decomposition import PCA
import os
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score

pca_obj = None

def dictionary(descriptors, N):
    '''
    Dictionary of SIFT features using GMM
    '''
    g = mixture.GaussianMixture(n_components=N)
    descriptors = np.asarray(descriptors)
    g.fit(descriptors)
    return g.means_, g.covariances_, g.weights_


def image_descriptors(file):
    '''
    Getting the SIFT descriptors of the image
    '''
    try:
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (256, 256))
        _, descriptors = cv2.xfeatures2d.SIFT_create(
            nfeatures=50).detectAndCompute(img, None)
        return descriptors
    except:
        print(file)


def folder_descriptors(folder):
    '''
    Getting all the SIFT image descriptions in a folder
    '''
    files = glob.glob(folder + "/*.png")
    print("Calculating descriptors. Number of images is", len(files))
    return np.concatenate([image_descriptors(file) for file in files if image_descriptors(file) is not None])


def likelihood_moment(x, ytk, moment):
    '''
    Calculating the likelihood moments
    '''
    x_moment = np.power(np.float32(
        x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk


def likelihood_statistics(samples, means, covs, weights):
    '''
    Calculating the likelihood statistics to build the FV
    '''
    gaussians, s0, s1, s2 = {}, {}, {}, {}

    g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular=False) for k in range(0, len(weights))]
    for index, x in enumerate(samples):
        gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in enumerate(samples):
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
    return s0, s1, s2


def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k] + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k]) for k in range(0, len(w))])


def normalize(fisher_vector):
    '''
    Power Normalization
    '''
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(samples, means, covs, w):
    '''
    Building the FV for a image, sample denotes a list of SIFT feature vectors
    '''
    global pca_obj
    samples = pca_obj.transform(samples)
    s0, s1, s2 = likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate(
        [np.concatenate(a), np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    # print("Fv")
    # print(fv)
    return np.array(fv)

def reduceDimensions(words):
    '''
    Using PCA to reduce dimensions
    '''
    global pca_obj
    if(pca_obj is None):
        pca = PCA(n_components=10)
        pca_obj = pca.fit(words)
    res = pca_obj.transform(words)
    return res


def generate_gmm(input_folder, N):
    '''
    Generating the GMM and saving the parameters
    '''
    words = np.concatenate([folder_descriptors(folder)
                            for folder in glob.glob(input_folder + '/*')])
    words = reduceDimensions(words)
    print("Training GMM of size", N)
    means, covs, weights = dictionary(words, N)
    # Throw away gaussians with weights that are too small:
    # th = 1.0 / N
    th = 0
    means = np.float32(
        [m for k, m in zip(range(0, len(weights)), means) if weights[k] > th])
    covs = np.float32(
        [m for k, m in zip(range(0, len(weights)), covs) if weights[k] > th])
    weights = np.float32(
        [m for k, m in zip(range(0, len(weights)), weights) if weights[k] > th])

    np.save("means.gmm", means)
    np.save("covs.gmm", covs)
    np.save("weights.gmm", weights)
    return means, covs, weights


def get_fisher_vectors_from_folder(folder, gmm):
    '''
    Getting the FVs of all the images in the folder
    '''
    files = glob.glob(folder + "/*.png")
    res = {}
    for file in files:
        temp = image_descriptors(file)
        if(temp is not None):
            # print(temp)
            # print(os.path.basename(file))
            res[os.path.basename(file)] = np.float32(fisher_vector(temp, *gmm))
    return res
    # return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])


def fisher_features(folder, gmm):
    '''
    Getting the FVs of all the images in the subfolders in the directory
    '''
    folders = glob.glob(folder + "/*")
    res = {}
    for f in folders:
        res.update(get_fisher_vectors_from_folder(f, gmm))
    return res


def train(gmm, features):
    '''
    Not used
    '''
    print(features)
    X = np.concatenate(features.values)
    Y = np.concatenate([np.float32([i]*len(v))
                        for i, v in zip(range(0, len(features)), features.values())])

    clf = svm.SVC()
    clf.fit(X, Y)
    return clf

def compare(features):
    pass

def success_rate(classifier, features):
    '''
    Not used
    '''
    print("Applying the classifier...")
    X = np.concatenate(np.array(features.values()))
    Y = np.concatenate([np.float32([i]*len(v))
                        for i, v in zip(range(0, len(features)), features.values())])
    res = float(
        sum([a == b for a, b in zip(classifier.predict(X), Y)])) / len(Y)
    return res


def load_gmm(folder=""):
    '''
    Not used
    '''
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    return map(lambda file: load(file), map(lambda s: folder + "/", files))


def get_args():
    '''
    Getting the command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', "--dir", help="Directory with images", default='.')
    parser.add_argument(
        '-dxml', "--dirxml", help="Directory with xml", default='.')
    parser.add_argument("-g", "--loadgmm", help="Load Gmm dictionary",
                        action='store_true', default=False)
    parser.add_argument(
        '-n', "--number", help="Number of words in dictionary", default=5, type=int)
    args = parser.parse_args()
    return args

def get_word_strings_from_file(file_path):
    '''
    Getting the word strings from the xml filepath
    '''
    res = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    lines = root.findall("./handwritten-part/line")
    for line in lines:
        for word in line.findall('word'):
            id = word.get('id')
            word_string = word.get('text')
            res[id+".png"] = word_string
    return res

def extractWordStrings(folder_path):
    '''
    Extracting the word strings from all the xml files present in the folder
    '''
    word_strings = {}
    folders = glob.glob(folder_path + "/*.xml")
    for file in folders:
        word_strings.update(get_word_strings_from_file(file))
    return word_strings    

def get_word_string(path):
    pass

def MAPScore(query_path, word_strings_dict, fisher_features, gmm):
    '''
    Getting the MAP score for the given image query
    '''
    query_sift_features = image_descriptors(query_path)
    query_FV = fisher_vector(query_sift_features, *gmm)
    query_FV = query_FV.reshape(1,-1)
    similarity_score = cosine_similarity(query_FV, np.array(list(fisher_features.values())))
    query_string = word_strings_dict[os.path.basename(query_path)]
    word_vals = np.array([word_strings_dict[your_key] for your_key in fisher_features.keys()])
    word_vals = word_vals.flatten()
    y_true = np.array([[int(1) if s == query_string else int(0) for s in word_vals]])
    map = label_ranking_average_precision_score(y_true, similarity_score)
    return map

if __name__ == '__main__':
    args = get_args()
    working_folder = args.dir
    print(working_folder)
    print("no. of weights {0}".format(args.number))
    gmm = load_gmm(working_folder) if args.loadgmm else generate_gmm(
        working_folder, args.number)
    fisher_features = fisher_features(working_folder, gmm)
    word_strings_dict = extractWordStrings(args.dirxml)
    scores = []
    while(True):
        query_path = input("Enter query image path: ")
        if(query_path == "break"):
            break
        score = MAPScore(query_path, word_strings_dict, fisher_features, gmm)
        scores.append(score)
        print(score)
