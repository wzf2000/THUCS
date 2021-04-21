#include <iostream>
#include <cstdio>

#define stature(p) ((p) ? (p)->height : -1)
#define IsRoot(x) (!((x).par))
#define IsLChild(x) (!IsRoot(x) && (&(x) == (x).par->ch[0]))
#define IsRChild(x) (!IsRoot(x) && (&(x) == (x).par->ch[1]))
#define FromParentTo(x) (IsRoot(x) ? this->_root : (IsLChild(x) ? (x).par->ch[0] : (x).par->ch[1]))
#define uncle(x) (IsLChild(*((x)->par)) ? (x)->par->par->ch[1] : (x)->par->par->ch[0])

#define IsBlack(p) (!(p) || (RB_BLACK == (p)->color))
#define IsRed(p) (!IsBlack(p))
#define BlackHeightUpdated(x) ( \
    (stature((x).ch[0]) == stature((x).ch[1])) && \
    ((x).height == (IsRed(&x) ? stature((x).ch[0]) : stature((x).ch[0]) + 1)) \
)

typedef enum { RB_RED, RB_BLACK } RBColor;

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
    int height;
    RBColor color;
    Node() : par(nullptr), ch({nullptr, nullptr}), height(0), color(RB_RED) {}
    Node(const T &e, Node *p = nullptr, Node *lc = nullptr, Node *rc = nullptr, int h = 0, RBColor c = RB_RED) : data(e), par(p), height(h), color(c)
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
    Node<T> *connect34(Node<T>*, Node<T>*, Node<T>*, Node<T>*, Node<T>*, Node<T>*, Node<T>*);
    Node<T> *rotateAt(Node<T> *x);
    virtual int updateHeight(Node<T> *x);
    virtual void updateHeightAbove(Node<T> *x);
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
int BST<T>::remove(Node<T> *x)
{
    FromParentTo(*x) = nullptr;
    updateHeightAbove(x->par);
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
int BST<T>::updateHeight(Node<T> *x)
{
    return x->height = 1 + max(stature(x->ch[0]), stature(x->ch[1]));
}

template<typename T>
void BST<T>::updateHeightAbove(Node<T> *x)
{
    for (; x; x = x->par)
        updateHeight(x);
}

template<typename T>
Node<T> *BST<T>::connect34(Node<T> *a, Node<T> *b, Node<T> *c, Node<T> *T0, Node<T> *T1, Node<T> *T2, Node<T> *T3)
{
    a->ch[0] = T0;
    if (T0) T0->par = a;
    a->ch[1] = T1;
    if (T1) T1->par = a;
    updateHeight(a);
    c->ch[0] = T2;
    if (T2) T2->par = c;
    c->ch[1] = T3;
    if (T3) T3->par = c;
    updateHeight(c);
    b->ch[0] = a;
    a->par = b;
    b->ch[1] = c;
    c->par = b;
    updateHeight(b);
    return b;
}

template<typename T>
Node<T> *BST<T>::rotateAt(Node<T> *v)
{
    Node<T> *p = v->par;
    Node<T> *g = p->par;
    if (IsLChild(*p))
    {
        if (IsLChild(*v))
        {
            p->par = g->par;
            return connect34(v, p, g, v->ch[0], v->ch[1], p->ch[1], g->ch[1]);
        }
        else
        {
            v->par = g->par;
            return connect34(p, v, g, p->ch[0], v->ch[0], v->ch[1], g->ch[1]);
        }
    }
    else
    {
        if (IsRChild(*v))
        {
            p->par = g->par;
            return connect34(g, p, v, g->ch[0], p->ch[0], v->ch[0], v->ch[1]);
        }
        else
        {
            v->par = g->par;
            return connect34(g, v, p, g->ch[0], v->ch[0], v->ch[1], p->ch[1]);
        }
    }
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
class RBT : public BST<T>
{
protected:
    void solveDoubleRed(Node<T> *x);
    void solveDoubleBlack(Node<T> *r);
    int updateHeight(Node<T> *x);
public:
    Node<T> *insert(const T&);
    bool remove(const T&);
};

template<typename T>
void RBT<T>::solveDoubleRed(Node<T> *x)
{
    if (IsRoot(*x))
    {
        this->_root->color = RB_BLACK;
        this->_root->height++;
        return;
    }
    Node<T> *p = x->par;
    if (IsBlack(p)) return;
    Node<T> *g = p->par;
    Node<T> *u = uncle(x);
    if (IsBlack(u))
    {
        if (IsLChild(*x) == IsLChild(*p))
            p->color = RB_BLACK;
        else
            x->color = RB_BLACK;
        g->color = RB_RED;
        Node<T> *gg = g->par;
        auto &ret = FromParentTo(*g);
        Node<T> *r = ret = this->rotateAt(x);
        r->par = gg;
    }
    else
    {
        p->color = RB_BLACK;
        p->height++;
        u->color = RB_BLACK;
        u->height++;
        if (!IsRoot(*g)) g->color = RB_RED;
        solveDoubleRed(g);
    }
}

template<typename T>
void RBT<T>::solveDoubleBlack(Node<T> *r)
{
    Node<T> *p = r ? r->par : this->_hot;
    if (!p) return;
    Node<T> *s = (r == p->ch[0]) ? p->ch[1] : p->ch[0];
    if (IsBlack(s))
    {
        Node<T> *t = nullptr;
        if (IsRed(s->ch[1])) t = s->ch[1];
        if (IsRed(s->ch[0])) t = s->ch[0];
        if (t)
        {
            RBColor oldColor = p->color;
            auto &ret = FromParentTo(*p);
            Node<T> *b = ret = this->rotateAt(t);
            if (b->ch[0])
            {
                b->ch[0]->color = RB_BLACK;
                updateHeight(b->ch[0]);
            }
            if (b->ch[1])
            {
                b->ch[1]->color = RB_BLACK;
                updateHeight(b->ch[1]);
            }
            b->color = oldColor;
            updateHeight(b);
        }
        else
        {
            s->color = RB_RED;
            s->height--;
            if (IsRed(p)) p->color = RB_BLACK;
            else
            {
                p->height--;
                solveDoubleBlack(p);
            }
        }
    }
    else
    {
        s->color = RB_BLACK;
        p->color = RB_RED;
        Node<T> *t = IsLChild(*s) ? s->ch[0] : s->ch[1];
        this->_hot = p;
        auto &ret = FromParentTo(*p);
        ret = this->rotateAt(t);
        solveDoubleBlack(r);
    }
}

template<typename T>
int RBT<T>::updateHeight(Node<T> *x)
{
    x->height = max(stature(x->ch[0]), stature(x->ch[1]));
    return IsBlack(x) ? x->height++ : x->height;
}

template<typename T>
Node<T> *RBT<T>::insert(const T &e)
{
    Node<T> *&x = this->search(e);
    if (x) return x;
    x = new Node<T>(e, this->_hot, nullptr, nullptr, -1);
    this->_size++;
    Node<T> *xOld = x;
    solveDoubleRed(x);
    return xOld;
}

template<typename T>
Node<T> *removeAt(Node<T> *&x, Node<T> *&hot)
{
    Node<T> *w = x;
    Node<T> *succ = nullptr;
    if (!x->ch[0]) succ = x = x->ch[1];
    else
        if (!x->ch[1]) succ = x = x->ch[0];
    else
    {
        w = w->succ();
        swap(x->data, w->data);
        Node<T> *u = w->par;
        ((u == x) ? u->ch[1] : u->ch[0]) = succ = w->ch[1];
    }
    hot = w->par;
    if (succ) succ->par = hot;
    delete w;
    return succ;
}

template<typename T>
bool RBT<T>::remove(const T &e)
{
    Node<T> *&x = this->search(e);
    if (!x) return false;
    Node<T> *r = removeAt(x, this->_hot);
    if (!(--this->_size)) return true;
    if (!this->_hot)
    {
        this->_root->color = RB_BLACK;
        updateHeight(this->_root);
        return true;
    }
    if (BlackHeightUpdated(*this->_hot)) return true;
    if (IsRed(r))
    {
        r->color = RB_BLACK;
        r->height++;
        return true;
    }
    solveDoubleBlack(r);
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
    RBT<int> tree;
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
    cerr << "Time cost by RBT is " << 1. * (clock() - time) / CLOCKS_PER_SEC << "s.\n";
    return 0;
}
