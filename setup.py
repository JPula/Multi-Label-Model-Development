# setup.py
'''
    The setup.py file stores and runs configurations for model creation, such as:
    preprocessing, feature vectorization, model blueprint, and evaluation variants.

    This setup file is currently configured to run: 
        - default preprocessing (conversion to lowercase, removal of special characters, filling Nan)
        - default Bag-of-Words Count Vectorization
        - default Multi-label Multinomial Naive Bayes Classifier
        - trained on a train partition from the Toxic Comment Classification train dataset
        - predicted a test partition from the Toxic Comment Classification train dataset
        - and evaluated with a simple accuracy score
'''


from scripts import utils, preprocessing, feature_selection, model_selection, model, evaluation

utils.reload_scripts()

DATA_DIR = 'data'

RAW_SUB_DIR = '00_raw'
PRO_SUB_DIR = '01_preprocessed'

train_df = None
test_df = None

# PREPROCESSING
RUN_PREPROCESSING = True

if RUN_PREPROCESSING:
    train_df = utils.load_data(DATA_DIR, RAW_SUB_DIR, 'train')
    test_df = utils.load_data(DATA_DIR, RAW_SUB_DIR, 'test')

    preprocessor = preprocessing.DefaultPreprocessor()

    processed_train_df = preprocessor.process(train_df)
    utils.export_data(processed_train_df, DATA_DIR, PRO_SUB_DIR, 'train')

    processed_test_df = preprocessor.process(test_df)
    utils.export_data(processed_train_df, DATA_DIR, PRO_SUB_DIR, 'test')
else:
    train_df = utils.load_data(DATA_DIR, PRO_SUB_DIR, 'train')
    test_df = utils.load_data(DATA_DIR, PRO_SUB_DIR, 'test')

# FEATURE SELECTION
X = train_df['comment_text']
X2_test = test_df['comment_text']
y = train_df.drop(['id', 'comment_text'], axis=1)

vectorizer = feature_selection.DefaultCountVectorizer()

X = vectorizer.fit_transform(X)
X2_test = vectorizer.fit_transform(X2_test)

# TRAIN SPLIT
X_train, X_test, y_train, y_test = model_selection.default_split(X, y)

# MODEL
clf = model.DefaultMultiNB()

# FIT
clf.fit(X_train, y_train)

# PREDICT
'''
    Clarification: "train-test" refers to the test partition taken from the Toxic Comment Classification train dataset,
                    while "test" refers to the Toxic Comment Classification train dataset. 
'''
PREDICT_ON = 'train-test'
predictions = None
    
if PREDICT_ON == 'train':    
    predictions = clf.predict(X_train)
elif PREDICT_ON == 'train-test':
    predictions = clf.predict(X_test)
elif PREDICT_ON == 'test':
    predictions = clf.predict(X2_test)

# EVALUATE
RUN_EVALUATION = True

if RUN_EVALUATION:
    evaluator = evaluation.DefaultScoring()

    if PREDICT_ON == 'train':
        evaluator.evaluate(y_train, predictions)
    elif PREDICT_ON == 'train-test':
        evaluator.evaluate(y_test, predictions)

# EXPORT PREDICTIONS

EXPORT_PREDICTIONS = True