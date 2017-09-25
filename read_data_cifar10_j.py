import numpy as np
import cPickle
import word_vec
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

embeddings = word_vec.get_vectors()

train_d = unpickle('cifar-10-batches-py/data_batch_1')
X = train_d['data']
Y = train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_2')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_3')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_4')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_5')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']

X = np.reshape(X, (50000,3,32,32)).transpose(0,2,3,1)

X_train = X[:40000]
Y_train = Y[:40000]
X_validation  = X[40000:]
Y_validation = Y[40000:]


def classification_baseline():
    import baseline
    enc = OneHotEncoder(sparse=False)
    Y_train = enc.fit_transform(np.array(Y_train).reshape(-1,1))
    Y_validation = enc.fit_transform(np.array(Y_validation).reshape(-1,1))
    baseline.train(X_train, Y_train, X_validation, Y_validation)

def classify_embedding():
    import classification_based
    class_labels = unpickle('cifar-10-batches-py/batches.meta')['label_names']
    # print class_labels
    Y_GO_train = np.array(Y_train)
    X_GO_train = np.array(X_train)

    #removed_indices = np.where(Y_8_train!=8)
    #Y_8_train = Y_8_train[removed_indices]
    #X_8_train = X_8_train[removed_indices]
    #removed_indices = np.where(Y_8_train!=9)
    #Y_8_train = Y_8_train[removed_indices]
    #X_8_train = X_8_train[removed_indices]

    enc = OneHotEncoder(sparse=False)
    Y_GO_train = enc.fit_transform(np.array(Y_GO_train).reshape(-1,1))

    Y_GO_validation = np.array(Y_validation)
    X_GO_validation = np.array(X_validation)

    #removed_indices = np.where(Y_8_validation!=8)
    #Y_8_validation = Y_8_validation[removed_indices]
    #X_8_validation = X_8_validation[removed_indices]
    #removed_indices = np.where(Y_8_validation!=9)
    #Y_8_validation = Y_8_validation[removed_indices]
    #X_8_validation = X_8_validation[removed_indices]

    Y_GO_validation = enc.fit_transform(np.array(Y_GO_validation).reshape(-1,1))

    classification_based.train(X_GO_train, Y_GO_train, X_GO_validation, Y_GO_validation)

    #Y_2_validation = np.array(Y_validation)
    #X_2_validation = np.array(X_validation)

    #indices = np.where(Y_2_validation>=8)
    #Y_2_validation = Y_2_validation[indices]
    #X_2_validation = X_2_validation[indices]

    validaiton_probab = classification_based.predict_probabilites(X_GO_validation)
    print validaiton_probab.shape
    print validaiton_probab[0]
    print Y_GO_validation[0]
    weights = []
    for i in class_labels:
        weights.append(embeddings[i])
    weights = np.array(weights, dtype=np.float32)

    validaiton_embeddings = np.dot(validaiton_probab, weights)
    print validaiton_embeddings.shape
    targets_embeddings = []
    targets_embeddings_words = []
    for i in class_labels:
        targets_embeddings.append(embeddings[i])
        targets_embeddings_words.append(i)

    for k in embeddings:
        if k not in class_labels:
            targets_embeddings.append(embeddings[k])
            targets_embeddings_words.append(k)

    targets_embeddings = np.array(targets_embeddings, dtype=np.float32)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from scipy.spatial.distance import cosine
    Y_pred_validation = []
    for i in validaiton_embeddings:
        cos = []
        for j in targets_embeddings:
            val = 1 - cosine(i,j)
            cos.append(val)

        #vec = np.zeros(len(targets_embeddings))
        #vec[np.argmax(cos)] = 1
        Y_pred_validation.append(np.argmax(cos))

    # neigh = KNeighborsClassifier(n_neighbors=1)
    # neigh.fit(targets_embeddings, [0,1,2,3,4,5,6,7,8,9])
    # Y_pred_validation = neigh.predict(validaiton_embeddings)
    # print Y_2_validation
    # print Y_pred_validation
    print accuracy_score(np.argmax(Y_GO_validation,axis=1), Y_pred_validation)

    #for i,j in zip(Y_2_validation, Y_pred_validation):
    #    print (i,j)

def regression_embedding():
    import regression_based
    class_labels = unpickle('cifar-10-batches-py/batches.meta')['label_names']
    #Y_8_train = np.array(Y_train)
    #X_8_train = np.array(X_train)
    #removed_indices = np.where(Y_8_train!=1)
    #Y_8_train = Y_8_train[removed_indices]
    #X_8_train = X_8_train[removed_indices]
    #removed_indices = np.where(Y_8_train!=4)
    #Y_8_train = Y_8_train[removed_indices]
    #X_8_train = X_8_train[removed_indices]
    Y_GO_train = np.array(Y_train)
    X_GO_train = np.array(X_train)
    Y_GO_test = np.array(Y_validation)
    X_GO_test = np.array(X_validation)
    #Y_8_train = [embeddings[class_labels[i]] for i in Y_8_train]
    Y_GO_train = [embeddings[class_labels[i]] for i in np.array(Y_GO_train)]

    #Y_8_validation = np.array(Y_validation)
    #X_8_validation = np.array(X_validation)


    regression_based.train(X_GO_train, Y_GO_train)

    #Y_2_validation = np.array(Y_validation)
    #X_2_validation = np.array(X_validation)

    #indices = np.where(np.logical_or(Y_2_validation==1 ,Y_2_validation==4))
    #Y_2_validation = Y_2_validation[indices]
    #X_2_validation = X_2_validation[indices]

    validaiton_embeddings = regression_based.predict(X_GO_test)

    targets_embeddings_words = []
    targets_embeddings = []
    for i in class_labels:
        targets_embeddings.append(embeddings[i])
        targets_embeddings_words.append(i)

    for k in embeddings:
        if k not in class_labels:
            targets_embeddings.append(embeddings[k])
            targets_embeddings_words.append(k)
    targets_embeddings = np.array(targets_embeddings, dtype=np.float32)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from scipy.spatial.distance import cosine
    # Y_pred_validation = []
    # for i in validaiton_embeddings:
    #     cos = []
    #     for j in targets_embeddings:
    #         val = cosine(i,j)
    #         cos.append(val)
    #     Y_pred_validation.append(np.argmax(cos))
    #
    neigh = KNeighborsClassifier(n_neighbors=1)
    print len(targets_embeddings)
    neigh.fit(targets_embeddings, range(0,len(targets_embeddings)))
    Y_pred_validation = neigh.predict([validaiton_embeddings[0]])
    print Y_pred_validation
    neighbors = neigh.kneighbors([validaiton_embeddings[0]],20,return_distance=False)
    print neighbors
    for n in neighbors[0]:
        print targets_embeddings_words[n]
    # print Y_2_validation
    # print Y_pred_validation
    #print accuracy_score([Y_GO_test[0]], Y_pred_validation)


    # data = np.vstack((validaiton_embeddings, targets_embeddings))
    # from sklearn import manifold
    # tsne = manifold.TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(data)
    #
    # print X_tsne.shape
    # import matplotlib.pyplot as plt
    # Y_2_validation = Y_pred_validation.tolist()  + [0,1,2,3,4,5,6,7,8,9]
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_2_validation, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 9.5)
    # plt.show()

#regression_embedding()
classify_embedding()
