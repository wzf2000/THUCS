#include <iostream>
#include <cstdio>

#define stature(p) ((p) ? (p)->height : -1)
#define IsRoot(x) (!((x).par))
#define IsLChild(x) (!IsRoot(x) && (&(x) == (x).par->ch[0]))
#define IsRChild(x) (!IsRoot(x) && (&(x) == (x).par->ch[1]))
#define FromParentTo(x) (IsRoot(x) ? this->_root : (IsLChild(x) ? (x).par->ch[0] : (x).par->ch[1]))
using namespace std;

int max(const int &a, const int &b)
{
    return a < b ? b : a;
}

template<typename T>
struct Node
{
    T data;
    Node<T> *par, *ch[2];
    Node() : par(nullptr), ch({nullptr, nullptr}) {}
    Node(const T &e, Node *p = nullptr, Node *lc = nullptr, Node *rc = nullptr) : data(e), par(p)
    {
        ch[0] = lc;
        ch[1] = rc;
    }
    Node<T>* succ();
    bool operator<(const Node &rhs) const
    {
        return data < rhs.data;
    }
    bool operator==(const Node &rhs) const
    {
        return data == rhs.data;
    }
};

template<typename T>
Node<T> *Node<T>::succ()
{
    Node<T> *s = this;
    if (ch[1])
    {
        s = ch[1];
        while (s->ch[0]) s = s->ch[0];
    }
    else
    {
        while (IsRChild(*s)) s = s->par;
        s = s->par;
    }
    return s;
}

template<typename T>
class BST
{
protected:
    int _size;
    Node<T> *_root, *_hot = nullptr;
public:
    BST() : _size(0), _root(nullptr) {}
    ~BST()
    {
        if (_size > 0) remove(_root);
    }
    int size() const
    {
        return _size;
    }
    bool empty() const
    {
        return !_root;
    }
    void dfs();
    void dfs(Node<T> *x);
    int remove(Node<T> *x);
    virtual Node<T> *&search(const T &e);
    virtual T searchMax(const T&e);
    virtual Node<T> *insert(const T &e) = 0;
    virtual bool remove(const T &e) = 0;
};

template<typename T>
int BST<T>::remove(Node<T> *x)
{
    FromParentTo(*x) = nullptr;
    int n = removeAt(x);
    _size -= n;
    return n;
}

template<typename T>
int removeAt(Node<T> *x)
{
    if (!x) return 0;
    int ret = 1 + removeAt(x->ch[0]) + removeAt(x->ch[1]);
    delete x;
    return ret;
}

template<typename T>
void BST<T>::dfs()
{
    dfs(_root);
}

template<typename T>
void BST<T>::dfs(Node<T> *x)
{
    if (x->ch[0] && x->ch[1])
        printf("%d: lc -> %d, rc -> %d\n", x->data, x->ch[0]->data, x->ch[1]->data);
    else
        if (x->ch[0])
            printf("%d: lc -> %d\n", x->data, x->ch[0]->data);
        else
            if (x->ch[1])
                printf("%d: rc -> %d\n", x->data, x->ch[1]->data);
            else
                printf("%d\n", x->data);
    if (x->ch[0]) dfs(x->ch[0]);
    if (x->ch[1]) dfs(x->ch[1]);
}

template<typename T>
Node<T> *&BST<T>::search(const T &e)
{
    if (!_root || e == _root->data)
    {
        _hot = nullptr;
        return _root;
    }
    for (_hot = _root; ; )
    {
        Node<T> *&c = (e < _hot->data) ? _hot->ch[0] : _hot->ch[1];
        if (!c || e == c->data) return c;
        _hot = c;
    }
}

template<typename T>
T BST<T>::searchMax(const T &e)
{
    T ret = -1;
    if (!_root) return ret;
    for (Node<T> *now = _root; ; )
    {
        if (e == now->data) return ret = e;
        if (now->data < e) ret = max(ret, now->data);
        Node<T> *&c = (e < now->data) ? now->ch[0] : now->ch[1];
        if (!c) return ret;
        now = c;
    }
}

template<typename T>
class Splay : public BST<T>
{
protected:
    Node<T> *splay(Node<T> *v);
    void rotate(Node<T> *v);
public:
    Node<T> *&search(const T&);
    T searchMax(const T&e);
    Node<T> *insert(const T&);
    bool remove(const T&);
};

template<typename T>
void Splay<T>::rotate(Node<T> *v)
{
    Node<T> *p = v->par, *g = p->par;
    bool k = IsRChild(*v);
    if (g) g->ch[IsRChild(*p)] = v;
    v->par = g;
    p->ch[k] = v->ch[!k];
    if (v->ch[!k]) v->ch[!k]->par = p;
    v->ch[!k] = p;
    p->par = v;
}

template<typename T>
Node<T> *Splay<T>::splay(Node<T> *v)
{
    if (!v) return nullptr;
    for (Node<T> *p = v->par; p; rotate(v), p = v->par)
        if (p->par) (IsLChild(*v) ^ IsLChild(*p)) ? rotate(v) : rotate(p);
    this->_root = v;
    return v;
}

template<typename T>
Node<T> *&Splay<T>::search(const T &e)
{
    Node<T> *p = BST<T>::search(e);
    this->_root = splay(p ? p : this->_hot);
    return this->_root;
}

template<typename T>
T Splay<T>::searchMax(const T &e)
{
    T ret = -1;
    if (!this->_root) return ret;
    for (Node<T> *now = this->_root; ; )
    {
        if (e == now->data)
        {
            splay(now);
            return ret = e;
        }
        if (now->data < e) ret = max(ret, now->data);
        Node<T> *&c = (e < now->data) ? now->ch[0] : now->ch[1];
        if (!c)
        {
            splay(now);
            return ret;
        }
        now = c;
    }
}

template<typename T>
Node<T> *Splay<T>::insert(const T &e)
{
    if (!this->_root)
    {
        this->_size++;
        return this->_root = new Node<T>(e);
    }
    if (e == search(e)->data) return this->_root;
    this->_size++;
    Node<T> *t = this->_root;
    if (this->_root->data < e)
    {
        t->par = this->_root = new Node<T>(e, nullptr, t, t->ch[1]);
        if (t->ch[1]) t->ch[1]->par = this->_root, t->ch[1] = nullptr;
    }
    else
    {
        t->par = this->_root = new Node<T>(e, nullptr, t->ch[0], t);
        if (t->ch[0]) t->ch[0]->par = this->_root, t->ch[0] = nullptr;
    }
    return this->_root;
}

template<typename T>
bool Splay<T>::remove(const T &e)
{
    if (!this->_root || (e != search(e)->data)) return false;
    Node<T> *w = this->_root;
    if (!this->_root->ch[0])
    {
        this->_root = this->_root->ch[1];
        if (this->_root) this->_root->par = nullptr;
    }
    else
        if (!this->_root->ch[1])
        {
            this->_root = this->_root->ch[0];
            if (this->_root) this->_root->par = nullptr;
        }
        else
        {
            Node<T> *l = this->_root->ch[0];
            l->par = nullptr;
            this->_root->ch[0] = nullptr;
            this->_root = this->_root->ch[1];
            this->_root->par = nullptr;
            search(w->data);
            this->_root->ch[0] = l;
            l->par = this->_root;
        }
    delete w;
    this->_size--;
    return true;
}

// Input optimization
inline int read()
{
    int f = 1;
    char ch;
    while (ch = getchar(), ch < '0' || ch > '9')
        if (ch == '-') f = -1;
    int x = ch - '0';
    while (ch = getchar(), ch >= '0' && ch <= '9')
        x = x * 10 + ch - '0';
    return x * f;
}

int main()
{
    int time = clock();
    int T = read();
    Splay<int> tree;
    while (T--)
    {
        char s[2];
        scanf("%s", s);
        int x = read();
        switch (s[0])
        {
            case 'A':
                tree.insert(x);
                break;
            case 'B':
                if (!tree.remove(x)) exit(1);
                break;
            case 'C':
                printf("%d\n", tree.searchMax(x));
                break;
        }
    }
    cerr << "Time cost by Splay is " << 1. * (clock() - time) / CLOCKS_PER_SEC << "s.\n";
    return 0;
}
