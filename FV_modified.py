
'''
shape of a (10,50)## 10 saamples,, 50 features
shape of gmeaans(4,50)
shape of gcovars(4,50)
shape of gpriors (4,)
gmeans, gcovars, gpriors, ll, pos = gmm(a, n_clusters=4, n_repetitions=100) 
Transpose means, covariances
convert gmeans, gcovars, to np.float32


shape of x x.shape(2number of samples, number of features...50)

convert x, to np.float32
ff = fisher(x.T, gmeans,gcovars,gpriors,  normalized=True, fast=True) 
shape of f (400, )
`k = 2 * the n_data_dimensions * n_components``.

'''


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
import timeit
from sklearn.metrics import label_ranking_average_precision_score
import pickle
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import phow


from cyvlfeat.gmm import gmm as GaussianMixture
from cyvlfeat.fisher import fisher as FisherVector

pca_obj = None
load_gmm_flag = False


def calcGaussian(descriptors_i):
    N=16
    gmeans, gcovars, gpriors, ll, pos = GaussianMixture(descriptors_i,n_clusters=N, max_num_iterations=100,n_repetitions=2,verbose=True)
    # g.fit(descriptors_i)
    # gweights = np.mean(gweights, axis=0)
    return (gmeans,gcovars,gpriors)

def dictionary(descriptors, desc_mapping, N):
    '''
    Dictionary of SIFT features using GMM
    '''
    means_ = []
    covariances_ = []
    weights_ = []

    pool = mp.Pool(mp.cpu_count())
    gmms = pool.map(calcGaussian, [np.asarray(descriptors[desc_mapping == i]) for i in range(max(desc_mapping)+1)])
    pool.close()
    means_ = [gmm[0] for gmm in gmms]
    covariances_ = [gmm[1] for gmm in gmms]
    weights_ = [gmm[2] for gmm in gmms]
    ###Parallel here
    # for i in range(max(desc_mapping)+1):
    #     g = mixture.GaussianMixture(n_components=N, max_iter=60)
    #     descriptors_i = np.asarray(descriptors[desc_mapping == i])
    #     g.fit(descriptors_i)
    #     means_.append(g.means_)
    #     covariances_.append(g.covariances_)
    #     weights_.append(g.weights_)
    return np.array(means_), np.array(covariances_), np.array(weights_)

def findKeyPoints(img, window=8, radius=16):
    '''
    Getting the key points using sliding window
    '''
    keyPoints = []
    for i in range(0,img.shape[0],window):
            for j in range(0,img.shape[1],window):
                keyPoints.append(cv2.KeyPoint(i,j,radius))
    return keyPoints

def findFeatures(gray, keypoints):
    '''
    Finding the intersting points using SIFT feature detector
    '''
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (256, 256))
    sift = cv2.xfeatures2d.SIFT_create()
    # print(len(keypoints))
    plt.imshow(gray)
    kp, des = sift.compute(gray, keypoints)
    return kp, des

def splitImage(im, M=2, N=6):
    im = im.copy()
    # print("image shape: {0}".format(im.shape))
    x_offset = math.ceil(im.shape[1]*1.00/N)
    y_offset = math.ceil(im.shape[0]*1.00/M)
    tiles = [im[y:min(y+y_offset,im.shape[0]),x:min(x+x_offset,im.shape[1])] for y in range(0,im.shape[0],y_offset)
                                                    for x in range(0,im.shape[1],x_offset)]
    return tiles

def getImgSegmentDescriptors(img1):
    '''
    Getting the interest points in the image segment
    '''
    # sizes = [2,4,6,8,10,12]
    sizes = [6]
    descriptors = None
    kp1, des1 = phow.vl_phow(img1, color="gray")
    kp1[:,1] = (kp1[:,1]-img1.shape[1]/2)/img1.shape[1]
    kp1[:,0] = (kp1[:,0]-img1.shape[0]/2)/img1.shape[0]
    print("descriptor shape: {0}, len of centers: {1}".format(des1.shape, len(kp1)))
    if(len(kp1) == 0):
        des1 = np.zeros((1,128))
        kp1 = np.zeros((1,2))
    des1 = np.concatenate((des1,kp1), axis=1)
    return des1

    # for i in range(len(windows)):
    #     # key_points_window = findKeyPoints(img1, windows[i], radius)
    #     # if(key_points_window is None):
    #     #     continue
    #     # kp1, des1 = findFeatures(img1, key_points_window)
    #     kp1, des1 = phow.vl_phow(img1, color="grey")
    #     if(len(kp1) == 0):
    #         continue
    #     # print("In Get segment descriptors, SIFT")
    #     # print(des1)
    #     # x,y = kp1[i].pt
    #     # x_enriched = (x-img1.shape[0]/2)/img1.shape[0]
    #     # y_enriched = (y-img1.shape[1]/2)/img1.shape[1]
    #     # print("descriptor shape: {0}, {1}".format(des1.shape, len(kp1)))
    #     # des1 = np.concatenate((des1,[[(point.pt[0]-img1.shape[0]/2)*1.00/img1.shape[0],
    #     #                     (point.pt[1]-img1.shape[1]/2)*1.00/img1.shape[1]] for point in kp1]), axis=1)
    #     # print(des1.shape)
    #     kp1[:,1] = (kp1[:,1]-img1.shape[1]/2)/img1.shape[1]
    #     kp1[:,0] = (kp1[:,0]-img1.shape[0]/2)/img1.shape[0]
    #     print("descriptor shape: {0}, {1}".format(des1.shape, len(kp1)))
    #     des1 = np.concatenate((des1,kp1), axis=1)
    #     print(des1.shape)
    #     if(descriptors is None):
    #         descriptors = des1
    #     np.concatenate((descriptors,des1),axis=0) 
    return descriptors

def image_descriptors(file):
    '''
    Computing the dense sift matching
    '''
    img1 = cv2.imread(file)
    img_segments = splitImage(img1, 2, 6)
    # print("No. of segments: {0}".format(len(img_segments)))
    mapping = []
    descriptors = None
    i = 0
    for img in img_segments:
        temp_descriptors = getImgSegmentDescriptors(img)
        if(descriptors is None):
            descriptors = temp_descriptors
        else:
            descriptors = np.concatenate((descriptors,temp_descriptors),axis=0)
        # print("fets shape, fets len: {0}, {1}".format(temp_descriptors.shape, len([i]*len(temp_descriptors))))
        mapping += [i]*len(temp_descriptors)
        # print(len(mapping), descriptors.shape)
        i = i + 1
    return (np.array(descriptors), np.array(mapping))

# def image_descriptors(file):
#     '''
#     Getting the SIFT descriptors of the image
#     '''
#     try:
#         img = cv2.imread(file, 0)
#         # img = cv2.resize(img, (256, 256))
#         # _, descriptors = cv2.xfeatures2d.SIFT_create(
#         #     nfeatures=50).detectAndCompute(img, None)
#         img = cv2.resize(img, (256, 256))
#         _, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
#         return descriptors
#     except:
#         print(file)


def folder_descriptors(folder):
    '''
    Getting all the SIFT image descriptions in a folder
    '''
    files = glob.glob(folder + "/*.png")
    print("Calculating descriptors. Number of images is", len(files))
    res = None
    mapping = None
    for file in files:
        img1 = cv2.imread(file)
        if(img1.shape[0] < 10 or img1.shape[1] < 10):
            continue
        # print("in folder desc-> getting desc for {0}".format(file))
        desc, temp_map = image_descriptors(file)
        # print(desc.shape, temp_map.shape)
        if desc is not None:
            if res is not None:
                res = np.concatenate((res,desc),axis=0)
                mapping = np.concatenate((mapping,temp_map),axis=0)
            else:
                res = desc
                mapping = temp_map
    print(res, mapping)
    return (res,mapping)


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

    g = [multivariate_normal(mean=means[k], cov=covs[k],
                             allow_singular=False) for k in range(0, len(weights))]
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
    Power and L2 Normalization
    '''
    v = np.multiply(np.sqrt(abs(fisher_vector)), np.sign(fisher_vector))
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(words_with_mapping, means, covs, w):
    '''
    Building the FV for a image, sample denotes a list of SIFT feature vectors
    '''
    # global pca_obj
    words = words_with_mapping[0]
    desc_mapping = words_with_mapping[1]
    words = reduceDimensions(words)
    # samples = pca_obj.transform(samples)
    fv = None
    for i in range(max(desc_mapping)+1):
        samples = np.asarray(words[desc_mapping == i])
        print("Sample shape ### ", samples.shape)
        samples = np.float32(samples.T)
        #samples = np.reshape(samples,(1,-1))
        means_i = means[i]
        covs_i = covs[i]
        w_i = w[i]

        means_i = means_i.T
        covs_i = covs_i.T
        means_i = np.float32(means_i)
        covs_i = np.float32(covs_i)
        w_i = np.float32(w_i)
        print("means shape: ", means_i.shape)
        print("covars shape : ", covs_i.shape)
        print("samples sjhape :" ,samples.shape)
        print("weights shape :  ", w_i.shape)
        fv_i = FisherVector(samples, means_i, covs_i, w_i, normalized=True, fast=True)
        # s0, s1, s2 = likelihood_statistics(samples, means_i, covs_i, w_i)
        # T = samples.shape[0]
        # covs_i = np.float32([np.diagonal(covs_i[k])
        #                 for k in range(0, covs_i.shape[0])])
        # a = fisher_vector_weights(s0, s1, s2, means_i, covs_i, w_i, T)
        # b = fisher_vector_means(s0, s1, s2, means_i, covs_i, w_i, T)
        # c = fisher_vector_sigma(s0, s1, s2, means_i, covs_i, w_i, T)
        # fv_i = np.concatenate(
        #     [np.concatenate(a), np.concatenate(b), np.concatenate(c)])
        if(fv is None):
            fv = fv_i
        else:
            fv = np.concatenate((fv,fv_i),axis = 0)
    fv = normalize(fv)
    return np.array(fv)


def reduceDimensions(words):
    '''
    Using PCA to reduce dimensions
    '''
    global pca_obj
    global load_gmm_flag
    # print(words.shape)
    # if(load_gmm_flag):
    #     with open("/home/praveen/Desktop/iiith-assignments/CV/project/35k_weights/pca_dump", 'rb') as handle:
    #         pca_obj = pickle.load(handle)
    # try:
    #     if(pca_obj is None):
    #         pca = PCA(n_components=62)
    #         pca_obj = pca.fit(words[:,:-2])
    #         with open("./pca_dump", 'wb') as handle:
    #             pickle.dump(pca_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     res = pca_obj.transform(words[:,:-2])
    #     res = np.concatenate((res,words[:,-2:]),axis=1)
    #     return res
    try:
        if(pca_obj is None):
            pca = PCA(n_components=64)
            pca_obj = pca.fit(words)
            with open("./pca_dump", 'wb') as handle:
                pickle.dump(pca_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res = pca_obj.transform(words)
        return res
    except:
        print("error in Reduce Dimensions")
        print("words shape: {0}".format(words.shape))


def loadPCA(path):
    global pca_obj
    with open(path + "/pca_dump", 'rb') as handle:
        pca_obj = pickle.load(handle)


def generate_gmm(input_folder, N):
    '''
    Generating the GMM and saving the parameters
    '''
    start = timeit.default_timer()
    ### Parallel
    pool = mp.Pool(mp.cpu_count())
    # words_with_mapping = [folder_descriptors(folder)
    #                         for folder in glob.glob(input_folder + '/*')]
    words_with_mapping = pool.map(folder_descriptors, [folder for folder in glob.glob(input_folder + '/*')])
    pool.close()
    words = np.concatenate([word[0] for word in words_with_mapping])
    word_mapping = np.concatenate([word[1] for word in words_with_mapping])
    print("shape of words: {0}, mapping: {1}".format(words.shape, word_mapping.shape))
    stop = timeit.default_timer()
    print('Time taken for getting features: ', stop - start)
    words = reduceDimensions(words)
    print("Training GMM of size", N)
    means, covs, weights = dictionary(words, word_mapping, N)
    #Throw away gaussians with weights that are too small:
    th = 1.0 / N
    th = 0
    for i in range(len(means)):
        means[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), means[i]) if weights[i][k] > th])
        covs[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), covs[i]) if weights[i][k] > th])
        weights[i] = np.float32(
            [m for k, m in zip(range(0, len(weights[i])), weights[i]) if weights[i][k] > th])
    np.save("means.gmm", means)
    np.save("covs.gmm", covs)
    np.save("weights.gmm", weights)
    return means, covs, weights


# # changed definition initial (folder, gmm)
def get_fisher_vectors_from_folder(gmm, folder):
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
            res[os.path.basename(file)] = np.float32(
                fisher_vector(temp, *gmm))
    return res
    # return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])


def fisher_features(folder, gmm):
    '''
    Getting the FVs of all the images in the subfolders in the directory
    '''
    folders = glob.glob(folder + "/*")
    res = {}
    def temp_fun(f, gmm):
        return get_fisher_vectors_from_folder(gmm, f)
    ## Parallel
    for f in folders:
        res.update(get_fisher_vectors_from_folder(gmm,f))
    # temp_fun = partial(get_fisher_vectors_from_folder, gmm)
    # pool = mp.Pool(mp.cpu_count())
    # results = pool.map(temp_fun, [f for f in folders])
    # pool.close()
    # for result in results:
    #     res.update(result)
    return res

def get_image_mapping_from_folder(folder):
    '''
    Getting the Image Name to absolute path mapping
    '''
    files = glob.glob(folder + "/*.png")
    res = {}
    for file in files:
        res[os.path.basename(file)] = os.path.abspath(file)
    return res
    # return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])


def get_image_mappings(folder):
    '''
    Getting the Image Name to absolute path mapping recursively
    '''
    folders = glob.glob(folder + "/*")
    res = {}
    for f in folders:
        res.update(get_image_mapping_from_folder(f))
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
    print("in load gmm")
    print(folder)
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    res = map(lambda file: np.load(file), map(
        lambda s: folder + "/" + s, files))
    # print(list(res))
    return res


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


def MAPScore(query_path, word_strings_dict, fisher_features, gmm, image_mapping_dict, show_img_flag = False):
    '''
    Getting the MAP score for the given image query
    '''
    if(show_img_flag):
        img = plt.imread(query_path)
        plt.imshow(img)
        plt.show()
    query_sift_features = image_descriptors(query_path)
    if(query_sift_features is None):
        return 0
    # print("path: {0}".format(query_path))
    # print(query_sift_features.shape)
    temp = copy.deepcopy(gmm)
    query_FV = fisher_vector(query_sift_features, *temp)
    # print(query_FV)
    query_FV = query_FV.reshape(1, -1)
    FV_values = np.array(list(fisher_features.values()))
    FV_keys = np.array(list(fisher_features.keys()))
    similarity_score = cosine_similarity(query_FV, FV_values)
    # print(similarity_score.shape)
    max_index = np.argmax(similarity_score)
    top_5_indices = similarity_score.flatten().argsort()[-5:][::-1]
    if(show_img_flag):
        print("top 5 indices {0}".format(top_5_indices))
        for i in top_5_indices:
            match_img_path = image_mapping_dict[FV_keys[i]]
            print("Matching image path: {0}".format(match_img_path))
            img = plt.imread(match_img_path)
            plt.imshow(img)
            plt.show()
    query_string = word_strings_dict[os.path.basename(query_path)]
    word_vals = np.array([word_strings_dict[your_key]
                          for your_key in fisher_features.keys()])
    word_vals = word_vals.flatten()
    y_true = np.array([[int(1) if s == query_string else int(0)
                        for s in word_vals]])
    mape = label_ranking_average_precision_score(y_true, similarity_score)
    return mape


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
        '-n', "--number", help="Number of words in dictionary", default=16, type=int)
    args = parser.parse_args()
    return args


####################################     Main     #####################################
# working_folder = "/home/praveen/Desktop/iiith-assignments/CV/project/kaggle_data_35k/a01"
# dir_xml = "/home/praveen/Desktop/iiith-assignments/CV/project/kaggle_data_35k/xml_testing/"
# load_folder = "/home/praveen/Desktop/iiith-assignments/CV/project/35k_weights"

# working_folder = "/home/praveen/Desktop/iiith-assignments/CV/project/kaggle_data_subset/a01"
# dir_xml = "/home/praveen/Desktop/iiith-assignments/CV/project/kaggle_data_subset/xml_testing/"
# load_folder = "/home/praveen/Desktop/iiith-assignments/CV/project/kaggle_data_subset/weights/"

working_folder = "kaggle_data_subset/a01"
dir_xml = "kaggle_data_subset/xml_testing/"
load_folder = "kaggle_data_subset/weights/"

print(working_folder)
no_gaussians = 16
print("no. of weights {0}".format(no_gaussians))
start = timeit.default_timer()
gmm = load_gmm(load_folder) if load_gmm_flag else generate_gmm(
    working_folder, no_gaussians)
stop = timeit.default_timer()
print('Time taken for training GMM: ', stop - start)
# print("gmm: {0}".format(gmm))

FV_features = None
if(load_gmm_flag):
    loadPCA(load_folder)
if(load_gmm_flag):
    with open(load_folder + "/FV_dump", 'rb') as handle:
        FV_features = pickle.load(handle)
else:
    FV_features = fisher_features(working_folder, gmm)
    with open("./FV_dump", 'wb') as handle:
        pickle.dump(fisher_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_strings_dict = None
if(load_gmm_flag):
    with open(load_folder + "/word_string_dict_dump", 'rb') as handle:
        word_strings_dict = pickle.load(handle)
else:
    word_strings_dict = extractWordStrings(dir_xml)
    with open("./word_string_dict_dump", 'wb') as handle:
        pickle.dump(word_strings_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
image_mapping_dict = get_image_mappings(working_folder)
scores = []
while(True):
    query_type = input(
        "Press 1 for test of multiple images\nPress 2 for single image\nPress 0 to exit\n")
    if(int(query_type) == 0):
        break
    if(int(query_type) == 1):
        score_list = []
        test_data_path = input("Enter query images folder path: ")
        folders = glob.glob(test_data_path + "/*")
        count = 0
        for folder in folders:
            image_paths = glob.glob(folder + "/*.png")
            for img_path in image_paths:
                count += 1
                # print("count: {0}".format(count))
                score = MAPScore(img_path, word_strings_dict,
                                 FV_features, gmm, image_mapping_dict, False)
                score_list.append(score)
        score_list = np.array(score_list)
        print("MAP Score: {0}".format(np.mean(score_list)))
    else:
        query_path = input("Enter query image path: ")
        if(query_path == "break"):
            break
        score = MAPScore(query_path, word_strings_dict,
                         FV_features, gmm, image_mapping_dict, True)
        scores.append(score)
        print("MAP Score: {0}".format(score))
