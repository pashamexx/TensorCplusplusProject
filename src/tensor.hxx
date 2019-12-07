#ifndef CPP_PROJECT_TENSOR_HXX
#define CPP_PROJECT_TENSOR_HXX

#include <iostream>
#include <fstream>

typedef int ARRAY_SIZE;
typedef int INT;
typedef float FLOAT;

char LOGGING = '0';
std::ofstream LOG_FILE("../logging.log", std::ios::out | std::ios::in);

// ====================================== N_DIM_ARRAY ==================================================================

// element of N-dimensional array that has a structure like a tree
// each element is pair (value, array of sons), where value is zero if array of sons != NULL
template<typename TYPE>
struct MemoryItem{
    TYPE value;
    MemoryItem *next;
};


// element for getting slice from tensor
// realises python-like slice list[start:end:step]
struct cutting{
    ARRAY_SIZE start = 0;
    ARRAY_SIZE end = 0;
    ARRAY_SIZE step = 1;
};


// ENABLE / DISABLE logging
void loggingEnable() { LOGGING = '1';}

void loggingDisable() { LOGGING = '0'; }

// ========================================= TENSOR ====================================================================

/*Numpy-like n-dimensional array*/

template<typename TYPE>
class Tensor{
public:
    // constructor of empty 0-dimensional array
    Tensor();

    /* Create n-dimension tensor of zeros with given shape
     *
     * @param dimension: number of dimensions (n)
     * @param shape: pointer on n-elements array - shape of tensor */
    Tensor(ARRAY_SIZE dimension, const ARRAY_SIZE *shape);

    /* Create 1-dimensional tensor from simple element
     *
     * @param element: any simple element*/
    explicit Tensor(TYPE element);

    /* Morph n1-dimension tensor with shape1 into n2-dimensional tensor with shape2 if it is possible.
     * product(shape1) == product(shape2) == amount of all the elements in tensor.
     *
     * @param new_dimension: n2
     * @param new_shape: shape2 */
    void reshape(ARRAY_SIZE new_dimension, const ARRAY_SIZE *new_shape);

    /* destructor - delete recursively tree-like array*/
    ~Tensor();


    ARRAY_SIZE dimension(){ return mDimension; };
    ARRAY_SIZE *shape(){
        auto res_shape = new ARRAY_SIZE[mDimension];
        for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){
            res_shape[itr] = mShape[itr];
        }
        return res_shape;
    };

    /* Returns the sequence of all the elements from tensor */
    TYPE* chain();

    /* Overload of operator =: create one more POINTER on tensor 'other' */
    Tensor<TYPE> &operator=(const Tensor<TYPE> &other);

    /* Overload of operator tensor[n] - returns i-th element of tensor (which is actually n-1-dimensional tensor) */
    Tensor<TYPE> &operator[](ARRAY_SIZE i);

    /* Overload of operator tensor[slice={start, end, step}] - returns python-like slice
     * of elements of tensor  (tensor[start:end:step] in python). */
    Tensor<TYPE> &operator[](cutting *slice);

    /* Overload of arithmetic operations with 2 tensors*/
    Tensor<TYPE> &operator+(const Tensor<TYPE> &other);
    Tensor<TYPE> &operator-(const Tensor<TYPE> &other);
    Tensor<TYPE> &operator*(const Tensor<TYPE> &other);
    Tensor<TYPE> &operator/(const Tensor<TYPE> &other);

    /* Overload of arithmetic operations with tensor and constant*/
    Tensor<TYPE> &operator+(TYPE singleElement);
    Tensor<TYPE> &operator-(TYPE singleElement);
    Tensor<TYPE> &operator*(TYPE singleElement);
    Tensor<TYPE> &operator/(TYPE singleElement);

    bool operator==(const Tensor<TYPE> &other);
    /* Create full copy of tensor */
    Tensor<TYPE> &copy();

    /* Print tensor in the standard output */
    void print();

    /* Print tensor in the given output */
    void write(std::ostream &stream);

    /* static functions for standard arithmetic operations */
    static TYPE sum(TYPE x, TYPE y){ return x + y; };
    static TYPE sub(TYPE x, TYPE y){ return x - y; };
    static TYPE mul(TYPE x, TYPE y){ return x * y; };
    static TYPE div(TYPE x, TYPE y){
        if (y == 0) throw std::invalid_argument("Divide by zero");
        return x / y;
    };

private:

    ARRAY_SIZE mDimension = 0;    // dimension of tensor
    MemoryItem<TYPE> *mArray;     // tree-like array of elements
    ARRAY_SIZE *mShape;           // pointer on the array of shapes


    /* Create n-dimensional tensor from given n-dimensional array
     *
     * @param array: pointer on array of MemoryItems which is actually recursive tree of elements.
     * @param dimension: number of dimensions (n)
     * @param shape: pointer on n-elements array - shape of tensor */
    Tensor(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape);


    /* Auxiliary recursive function for creating n-dimensional tree-like array of zeros.
     *
     * @param dimension: dimension of the given array (depth of tree)
     * @param shape: array of all the shape of tree
     * @param dimLevel: start index of elements from shape, that relate to given Level of array*/
    MemoryItem<TYPE> *createNDimArray(ARRAY_SIZE dimension, const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel);


    /* Create n-dimensional tree-like array of zeros
     *
     * @param dimension: dimension of the given array (depth of tree)
     * @param shape: pointer on the array of shapes of tree.*/
    MemoryItem<TYPE> *createNDimArray(ARRAY_SIZE dimension, const ARRAY_SIZE *shape);


    /* Create full copy of tree-like array
     *
     * @param array: root of the tree
     * @param dimension: depth of the tree
     * @param shape: all the shape of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root*/
    MemoryItem<TYPE> *copyNDimArray(MemoryItem<TYPE> *array,
                                    ARRAY_SIZE dimension,
                                    const ARRAY_SIZE *shape,
                                    ARRAY_SIZE dimLevel);


    /* Auxiliary recursive function for deleting n-dimensional tree-like array
     *
     * @param array: root of the tree
     * @param dimension: depth of the tree
     * @param shape: all the shape of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root*/
    void deleteNDimArray(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel);


    /* Delete n-dimensional tree-like array
     *
     * @param array: root of the tree
     * @param dimension: depth of the tree
     * @param shape: all the shape of the tree */
    void deleteNDimArray(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape);


    /* Auxiliary recursive function for creating the sequence of all the elements of tree-like array
     *
     * @param array: root of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root
     * @param resArray: pointer on the result array
     * @param startIndex: start position of the free space in resArray */
    void createChain(MemoryItem<TYPE> *array, ARRAY_SIZE dimLevel, TYPE *resArray, ARRAY_SIZE &startIndex);


    /* Auxiliary recursive function for reshaping of the tensor.
     * Fills the tensor with elements from the given sequence
     *
     * @param array: root of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root
     * @param _chain: result of the method .chain() - sequence of all the elements of the tensor
     * @param startIndex: start position of reading elements from _chain*/
    void fillArrayFromChain(MemoryItem<TYPE> *array, ARRAY_SIZE dimLevel, TYPE *_chain,
                            ARRAY_SIZE &startIndex);


    /* Auxiliary recursive function for printing the array.
     *
     * @param stream: output stream
     * @param array: root of the tree
     * @param dimension: depth of the tree
     * @param shape: all the shapes of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root */
    void printNDimArray(std::ostream &stream, MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape,
            ARRAY_SIZE dimLevel);


    /* Auxiliary recursive function for copying the array
     *
     * @param array1: root of array-to
     * @param array2: root of array-from
     * @param dimension: depth of the tree
     * @param shape: all the shapes of the tree
     * @param dimLevel: start index of elements from shape, that relate to given root */
    void copyValues(MemoryItem<TYPE> *array1, MemoryItem<TYPE> *array2, ARRAY_SIZE dimension, const ARRAY_SIZE *shape,
                    ARRAY_SIZE dimLevel);


    /* Auxiliary recursive function for realisation of operations between tensor and constant
     *
     * @param ptr2Func: pointer to function from 2 arguments (first argument - element of tensor, second - constant)
     * @param array: root of tree of tensor
     * @param singleElement: constant;
     * @param resArray: root of tree of the result tensor
     * @param dimension: depth of the tree
     * @param shape: all the shapes of the tree
     * @param dimLevel: start index of elements from shape, that relate to given rootl */
    void doUnaryOperation(TYPE (*ptr2Func)(TYPE , TYPE ), MemoryItem<TYPE> *array, TYPE singleElement,
            MemoryItem<TYPE> *resArray, ARRAY_SIZE dimension, const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel);


    /* Auxiliary recursive function for realisation of operations between two tensors
     *
     * @param ptr2Func: pointer to function from 2 arguments (1-rst argument - element of 1-rst tensor, 2-nd - of 2-nd)
     * @param array1: root of tree of 1-rst tensor
     * @param array2: root of tree of 2-nd tensor
     * @param resArray: root of tree of the result tensor
     * @param dimension: depth of the tree
     * @param shape: all the shapes of the tree
     * @param dimLevel: start index of elements from shape, that relate to given rootl */
    void doBinaryOperation(TYPE (*ptr2Func)(TYPE , TYPE ), MemoryItem<TYPE> *array1,
            MemoryItem<TYPE> *array2, MemoryItem<TYPE> *resArray, ARRAY_SIZE dimensional, const ARRAY_SIZE *shape,
            ARRAY_SIZE dimLevel);

    bool isEqual(MemoryItem<TYPE> *array1, MemoryItem<TYPE> *array2, ARRAY_SIZE dimensional, const ARRAY_SIZE *shape,
            ARRAY_SIZE dimLevel);
};


// ====================================== N_DIM_ARRAY ==================================================================

// recursive function to create n dimensional array like tree
template<typename TYPE>
MemoryItem<TYPE>* Tensor<TYPE>::createNDimArray(ARRAY_SIZE dimension, const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel){
    if (LOGGING == '1')
        LOG_FILE << "======createNDimArray (addition)=====(" << dimension << ' ' << dimLevel << ')' << std::endl;
    if (dimension < 1){
        return nullptr;
    } else {
        auto *res = new MemoryItem<TYPE>[shape[dimLevel]];
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            res[itr] = MemoryItem<TYPE>();
            if (dimension == 1) {
                res[itr].next = nullptr;
            } else {
                res[itr].next = createNDimArray(dimension-1, shape, dimLevel+1);
            }
            res[itr].value = 0;
        }
        return res;
    }
}


// create n dimension array like tree
template<typename TYPE>
MemoryItem<TYPE>* Tensor<TYPE>::createNDimArray(ARRAY_SIZE dimension, const ARRAY_SIZE *shape){
    if (LOGGING == '1') LOG_FILE << "======createNDimArray=====(" << dimension << ')' << std::endl;
    return createNDimArray(dimension, shape, 0);
}


// delete n-dimensional array
template <typename TYPE>
void Tensor<TYPE>::deleteNDimArray(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape,
        ARRAY_SIZE dimLevel){
    if (LOGGING == '1') LOG_FILE << "======deleteNDimArray=====(" << dimension << ')' << std::endl;
    if (dimension > 1){
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            deleteNDimArray(array[itr].next, dimension-1, shape, dimLevel+1);
        }

    }
    delete[] array;
}


// full copy of n-dimensional array
template <typename TYPE>
MemoryItem<TYPE>* Tensor<TYPE>::copyNDimArray(MemoryItem<TYPE> *array,ARRAY_SIZE dimension, const ARRAY_SIZE *shape,
                                              ARRAY_SIZE dimLevel) {
    if (LOGGING == '1') LOG_FILE << "======copyNDimArray=====(" << dimension << ')' << std::endl;
    if (dimension < 1){
        return nullptr;
    }
    auto *res = new MemoryItem<TYPE>[shape[dimLevel]];
    for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
        MemoryItem<TYPE> tmp;
        if (dimension > 1){
            tmp.next = copyNDimArray(array[itr].next, dimension-1, shape, dimLevel + 1);
            tmp.value = 0;
        } else {
            tmp.value = array[itr].value;
            tmp.next = nullptr;
        }
        res[itr] = tmp;
    }
    return res;
}

// non-recursive function for deleting
template <typename TYPE>
void Tensor<TYPE>::deleteNDimArray(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape) {
    deleteNDimArray(array, dimension, shape, 0);
}


// =========================================== TENSOR ==================================================================

// constructor of an empty array
template<typename TYPE>
Tensor<TYPE>::Tensor()
        :mShape(nullptr)
        ,mArray(nullptr)
{
    if (LOGGING == '1') LOG_FILE << "========create empty tensor=========" << std::endl;
}

// constructor from single element
template<typename TYPE>
Tensor<TYPE>::Tensor(TYPE element)
        :mShape(nullptr)
        ,mArray(nullptr)
{
    if (LOGGING == '1') LOG_FILE << "========create tensor=========(" << element << ')' << std::endl;
    mShape = new ARRAY_SIZE[1];
    mShape[0] = 1;
    mArray = new MemoryItem<TYPE>[1];
    mArray[0].next = nullptr;
    mArray[0].value = element;
    mDimension = 1;
}

// constructor of an n-dimension array (m1, m2, m3, ...) of zeros
template<typename TYPE>
Tensor<TYPE>::Tensor(ARRAY_SIZE dimension, const ARRAY_SIZE *shape)
        :mShape(nullptr)
        ,mArray(nullptr)
{
    if (LOGGING == '1') LOG_FILE << "========create tensor=========(dim: " << dimension << ')' << std::endl;
    mDimension = dimension;
    mShape = new ARRAY_SIZE[dimension];
    for (ARRAY_SIZE itr = 0; itr < dimension; ++itr){
        mShape[itr] = shape[itr];
    }
    mArray = createNDimArray(dimension, shape);
}


// constructor from n-dimensional array
template<typename TYPE>
Tensor<TYPE>::Tensor(MemoryItem<TYPE> *array, ARRAY_SIZE dimension, const ARRAY_SIZE *shape)
        :mShape(nullptr)
        ,mArray(nullptr)
{
    if (LOGGING == '1') LOG_FILE << "========create tensor from array=========()" << std::endl;
    mDimension = dimension;
    mShape = new ARRAY_SIZE[dimension];
    for (ARRAY_SIZE itr = 0; itr < dimension; ++itr){
        mShape[itr] = shape[itr];
    }
    mArray = array;
}

// destructor
template <typename TYPE>
Tensor<TYPE>::~Tensor() {
    if (LOGGING == '1') LOG_FILE << "========delete tensor=========(dim:" << mDimension << ')' << std::endl;
    deleteNDimArray(mArray, mDimension, mShape);
    delete[] mShape;
    mDimension = 0;
}


// recursively fill the chain array
template<typename TYPE>
void Tensor<TYPE>::createChain(MemoryItem<TYPE> *array, ARRAY_SIZE dimLevel, TYPE *resArray, ARRAY_SIZE &startIndex) {
    if (LOGGING == '1') LOG_FILE << "========create chain of tensor=========(" << dimLevel << ')' << std::endl;
    if (mDimension - dimLevel < 1){
        return;
    } else if (mDimension - dimLevel == 1){
        for (ARRAY_SIZE itr = 0; itr < mShape[dimLevel]; ++itr){
            resArray[startIndex] = array[itr].value;
            startIndex++;
        }
    } else {
        for (ARRAY_SIZE itr = 0; itr < mShape[dimLevel]; ++itr){
            createChain(array[itr].next, dimLevel + 1, resArray, startIndex);
        }
    }
}


// build a chain of all the elements of Array
template<typename TYPE>
TYPE* Tensor<TYPE>::chain() {
    if (LOGGING == '1') LOG_FILE << "========chain of tensor=========(" << mDimension << ')' << std::endl;
    ARRAY_SIZE amount = 1;
    for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){
        amount *= mShape[itr];
    }
    if (mDimension >= 1){
        TYPE *resArray = new TYPE[amount];
        ARRAY_SIZE startIndex = 0;
        createChain(mArray, 0, resArray, startIndex);
        return resArray;
    } else {
        TYPE *resArray = nullptr;
        return resArray;
    }
}

// fills n dimensional array from 1-dimensional chain
template <typename TYPE>
void Tensor<TYPE>::fillArrayFromChain(MemoryItem<TYPE> *array, ARRAY_SIZE dimLevel, TYPE *_chain,
                                      ARRAY_SIZE &startIndex) {
    if (LOGGING == '1') LOG_FILE << "========create tensor from chain=========(" << dimLevel << ')' << std::endl;
    if (mDimension - dimLevel < 1){
        return;
    } else if (mDimension - dimLevel == 1){
        for (ARRAY_SIZE itr = 0; itr < mShape[dimLevel]; ++itr){
            array[itr].value = _chain[startIndex];
            startIndex++;
        } return;
    } else {
        for (ARRAY_SIZE itr = 0; itr < mShape[dimLevel]; ++itr){
            fillArrayFromChain(array[itr].next, dimLevel + 1, _chain, startIndex);
        }
        return;
    }

}


// reshape array to new measurements
template<typename TYPE>
void Tensor<TYPE>::reshape(ARRAY_SIZE new_dimension, const ARRAY_SIZE *new_shape) {
    if (LOGGING == '1')
        LOG_FILE << "========reshape=========(from:" << mDimension << " to:" << new_dimension << ')' << std::endl;
    ARRAY_SIZE amount1 = 1;
    ARRAY_SIZE amount2 = 1;

    for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){    // check if we can do reshape
        amount1 *= mShape[itr];
    }
    for (ARRAY_SIZE itr = 0; itr < new_dimension; ++itr){
        amount2 *= new_shape[itr];
    }
    if (amount1 != amount2){
        throw std::length_error("Bad dimension for reshape");
    }

    TYPE *_chain = chain();

    MemoryItem<TYPE> *newArray = createNDimArray(new_dimension, new_shape);
    mDimension = new_dimension;
    mShape = new ARRAY_SIZE[new_dimension];
    for (ARRAY_SIZE itr = 0; itr < new_dimension; ++itr){
        mShape[itr] = new_shape[itr];
    }
    ARRAY_SIZE startIndex = 0;
    fillArrayFromChain(newArray, 0, _chain, startIndex);
    mArray = newArray;
}

// full copy of tensor
template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::copy() {
    if (LOGGING == '1') LOG_FILE << "========copy tensor=========(" << ')' << std::endl;
    auto *res = new Tensor<TYPE>(mDimension, mShape);
    res->mArray = copyNDimArray(mArray, mDimension, mShape, 0);
    return *res;
}

// x1 = x2 copying all the values from x2 to x1
template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator=(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator = =========(" << ')' << std::endl;
    if (this != &other){
        if (this->mDimension != other.mDimension) throw std::invalid_argument("different dimensions");
        bool success = true;
        for (ARRAY_SIZE itr = 0; itr < this->mDimension; ++itr)
            if (this->mShape[itr] != other.mShape[itr]){
                success = false;
                break;
            }
        if (!success) throw std::invalid_argument("different shapes");

        copyValues(this->mArray, other.mArray, mDimension, mShape, 0);
    }
    return *this;
}

// returns n-1 dimensional element of tensor (even if this element just simple value)
template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator[](ARRAY_SIZE i) {
    if (LOGGING == '1') LOG_FILE << "========operator[] =========(" << i << ')' << std::endl;
    if (i > mShape[0] || i < 0){
        throw std::out_of_range("index is out of range");
    } else if (mDimension > 1){
        auto *newShape = new ARRAY_SIZE[mDimension - 1];
        for (ARRAY_SIZE itr = 0; itr < mDimension - 1; ++itr){
            newShape[itr] = mShape[itr+1];
        }
        auto *res = new Tensor<TYPE>(mArray[i].next, mDimension - 1, newShape);
        return *res;
    } else {Tensor<TYPE>();
        auto *tmpArr = &mArray[i];
        auto *tmpShape = new ARRAY_SIZE[1];
        tmpShape[0] = 1;
        auto *res = new Tensor<TYPE>(tmpArr, 1, tmpShape);
        return *res;
    }
}

template<typename TYPE>
void Tensor<TYPE>::print() {
    if (LOGGING == '1') LOG_FILE << "========print tensor=========(" << ')' << std::endl;
    printNDimArray(std::cout, mArray, mDimension, mShape, 0);
    std::cout << std::endl;
}

template<typename TYPE>
void Tensor<TYPE>::write(std::ostream &stream){
    if (LOGGING == '1') LOG_FILE << "========write tensor to file =========(" << ')' << std::endl;
    printNDimArray(stream, mArray, mDimension, mShape, 0);
    stream << std::endl;
}

template<typename TYPE>
void Tensor<TYPE>::printNDimArray(std::ostream &stream, MemoryItem<TYPE> *array, ARRAY_SIZE dimension,
                                  const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel) {
    if (LOGGING == '1') LOG_FILE << "========print array =========(dim:" << dimension << ')' << std::endl;
    if (dimension == 1){
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            stream << array[itr].value << ' ';
        }
        stream << std::endl;
    } else {
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            printNDimArray(stream, array[itr].next, dimension-1, shape, dimLevel+1);
            for (ARRAY_SIZE i = 0; i < dimension-2; i++){
                stream << std::endl;
            }
        }
    }
}

template <typename TYPE>
void Tensor<TYPE>::copyValues(MemoryItem<TYPE> *array1, MemoryItem<TYPE> *array2, ARRAY_SIZE dimension,
                              const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel) {
    if (LOGGING == '1') LOG_FILE << "========copy values of array=========(" << ')' << std::endl;
    if (dimension == 1){
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            array1[itr].value = array2[itr].value;
        }
    } else {
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr) {
            copyValues(array1[itr].next, array2[itr].next, dimension - 1, shape, dimLevel + 1);
        }
    }
}

template <typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator[](cutting *slice) {
    if (LOGGING == '1'){
        LOG_FILE << "========operator [] for slice =========(" <<slice->start << ':' << slice->end;
        LOG_FILE << ':' << slice->step << ')' << std::endl;
    }
    if (slice->start < slice->end && slice->step > 0){
        ARRAY_SIZE amount = (slice->end - slice->start) / slice->step;
        if (slice->step != 1) amount++;
        auto *tmpArray = new MemoryItem<TYPE>[amount];
        ARRAY_SIZE tmpIndex = 0;
        for (ARRAY_SIZE itr = slice->start; itr < mShape[0] && itr < slice->end && itr >= 0; itr += slice->step){
            tmpArray[tmpIndex] = mArray[itr];
            tmpIndex += 1;
        }
        ARRAY_SIZE *resShape;
        resShape = new ARRAY_SIZE[mDimension];
        for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){
            resShape[itr] = mShape[itr];
        }
        resShape[0] = amount;
        // create *res instead res because of deleting local variables
        auto *res = new Tensor<TYPE>(tmpArray, mDimension, resShape);
        return *res;
    } else if (slice->end < slice->start && slice->step < 0){
        ARRAY_SIZE amount = (slice->start - slice->end) / -slice->step;
        if (slice->step != -1) amount++;
        auto *tmpArray = new MemoryItem<TYPE>[amount];
        ARRAY_SIZE j = 0;
        for (ARRAY_SIZE itr = slice->start; itr > slice->end && itr >= 0 && itr < mShape[0]; itr += slice->step){
            tmpArray[j] = mArray[itr];
            j++;
        }
        auto *tmpShape = new ARRAY_SIZE[mDimension];
        for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){
            tmpShape[itr] = mShape[itr];
        }
        tmpShape[0] = amount;
        // create *res instead res because of deleting local variables
        auto *res = new Tensor<TYPE>(tmpArray, mDimension, tmpShape);
        return *res;
    } else {
        auto res = Tensor<TYPE>();
        return res;
    }
}

// ---------------------------------------ARITHMETIC OPERATORS ---------------------------------------------------------
template <typename TYPE>
void Tensor<TYPE>::doUnaryOperation(TYPE (*ptr2Func)(TYPE ,TYPE ), MemoryItem<TYPE> *array, TYPE singleElement,
        MemoryItem<TYPE> *resArray, ARRAY_SIZE dimension, const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel) {
    if (LOGGING == '1') LOG_FILE << "========doUnaryOperation=========(" << ')' << std::endl;
    if (dimension == 1){
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            resArray[itr].value = (*ptr2Func)(array[itr].value, singleElement);
        }
    } else {
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            doUnaryOperation(ptr2Func, array[itr].next, singleElement, resArray[itr].next, dimension-1, shape,
                    dimLevel + 1);
        }
    }
}


template <typename TYPE>
void Tensor<TYPE>::doBinaryOperation(TYPE (*ptr2Func)(TYPE , TYPE ), MemoryItem<TYPE> *array1,
        MemoryItem<TYPE> *array2, MemoryItem<TYPE> *resArray, ARRAY_SIZE dimensional, const ARRAY_SIZE *shape,
        ARRAY_SIZE dimLevel) {

    if (LOGGING == '1') LOG_FILE << "========doBinaryOperation=========(" << ')' << std::endl;
    if (dimensional == 1){
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            resArray[itr].value = (*ptr2Func)(array1[itr].value, array2[itr].value);
        }
    } else {
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            doBinaryOperation(ptr2Func, array1[itr].next, array2[itr].next, resArray[itr].next, dimensional-1,
                    shape, dimLevel + 1);
        }
    }
}


static void checkSame(ARRAY_SIZE dim1, ARRAY_SIZE dim2, const ARRAY_SIZE *shape1, const ARRAY_SIZE *shape2){
    if (dim1 != dim2)
        throw std::invalid_argument("Can't find sum of tensors with different dimensions");
    if (LOGGING == '1') LOG_FILE << "=========check sames of arrays =========(" << ')' << std::endl;
    bool succ = true;
    for (ARRAY_SIZE itr = 0; itr < dim1; ++itr)
        if (shape1[itr] != shape2[itr]){
            succ = false;
            break;
        }
    if (!succ)
        throw std::invalid_argument("Can't find sum of tensors with different shape");

}

template <typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator+(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator + for other =========(" << ')' << std::endl;
    checkSame(this->mDimension, other.mDimension, this->mShape, other.mShape);
    auto *result = new Tensor<TYPE>(mDimension, mShape);
    doBinaryOperation(&Tensor<TYPE>::sum, this->mArray, other.mArray, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}


template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator-(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator - for other =========(" << ')' << std::endl;
    checkSame(this->mDimension, other.mDimension, this->mShape, other.mShape);
    auto *result = new Tensor<TYPE>(mDimension, mShape);
    doBinaryOperation(&Tensor<TYPE>::sub, this->mArray, other.mArray, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}

template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator*(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator * for other =========(" << ')' << std::endl;
    checkSame(this->mDimension, other.mDimension, this->mShape, other.mShape);
    auto *result = new Tensor<TYPE>(mDimension, mShape);
    doBinaryOperation(&Tensor<TYPE>::mul, this->mArray, other.mArray, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}

template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator/(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator / for other =========(" << ')' << std::endl;
    checkSame(this->mDimension, other.mDimension, this->mShape, other.mShape);
    auto *result = new Tensor<TYPE>(mDimension, mShape);
    doBinaryOperation(&Tensor<TYPE>::div, this->mArray, other.mArray, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}


template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator+(TYPE singleElement) {
    if (LOGGING == '1') LOG_FILE << "========operator + for single element =========(" << ')' << std::endl;
    auto *result = new Tensor<TYPE>(mDimension, mShape);

    doUnaryOperation(Tensor<TYPE>::sum, this->mArray, singleElement, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}


template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator-(TYPE singleElement) {
    if (LOGGING == '1') LOG_FILE << "========operator - for single element =========(" << ')' << std::endl;
    auto *result = new Tensor<TYPE>(mDimension, mShape);

    doUnaryOperation(Tensor<TYPE>::sub, this->mArray, singleElement, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}


template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator*(TYPE singleElement) {
    if (LOGGING == '1') LOG_FILE << "========operator * for single element =========(" << ')' << std::endl;
    auto *result = new Tensor<TYPE>(mDimension, mShape);

    doUnaryOperation(Tensor<TYPE>::mul, this->mArray, singleElement, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}


template<typename TYPE>
Tensor<TYPE>& Tensor<TYPE>::operator/(TYPE singleElement) {
    if (LOGGING == '1') LOG_FILE << "========operator / for single element =========(" << ')' << std::endl;
    auto *result = new Tensor<TYPE>(mDimension, mShape);

    doUnaryOperation(Tensor<TYPE>::div, this->mArray, singleElement, result->mArray, this->mDimension, this->mShape, 0);
    return *result;
}

template <typename TYPE>
bool Tensor<TYPE>::operator==(const Tensor<TYPE> &other) {
    if (LOGGING == '1') LOG_FILE << "========operator ==  =========(" << ')' << std::endl;
    bool success = this->mDimension == other.mDimension;

    if (!success) return false;

    for (ARRAY_SIZE itr = 0; itr < mDimension; ++itr){
        if (this->mShape[itr] != other.mShape[itr]){
            return false;
        }
    }

    success = isEqual(this->mArray, other.mArray, mDimension, mShape, 0);
    return success;
}


template <typename TYPE>
bool Tensor<TYPE>::isEqual(MemoryItem<TYPE> *array1, MemoryItem<TYPE> *array2, ARRAY_SIZE dimensional,
                           const ARRAY_SIZE *shape, ARRAY_SIZE dimLevel) {
    if (LOGGING == '1') LOG_FILE << "========are arrays equal=========(" << ')' << std::endl;
    if (dimensional == 1) return array1[0].value == array2[0].value;
    else {
        bool success = true;
        for (ARRAY_SIZE itr = 0; itr < shape[dimLevel]; ++itr){
            success &= isEqual(array1[itr].next, array2[itr].next, dimensional-1, shape, dimLevel+1);
            if (!success) break;
        }
    return success;
    }
}

#endif //CPP_PROJECT_TENSOR_HXX