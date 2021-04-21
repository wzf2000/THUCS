#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<iostream>

#define DEFAULT_CAPACITY 1000

using namespace std;

template<typename T>
class myVector {
public:
	int _size;
	int _capacity;
	T* _elem;
	myVector(int s = DEFAULT_CAPACITY)
	{
		_elem = new T[_capacity = s];
		_size = 0;
	}
	myVector(const T* A, int lo, int hi)
	{
		if (lo >= hi)
		{
			_elem = new T[_capacity = DEFAULT_CAPACITY];
			_size = 0;
		}
		else
			copyFrom(A, lo, hi);
	}
	void copyFrom(const T* A, int lo, int hi)	//[lo,hi)
	{
		_elem = new T[_capacity = 2 * (hi - lo)];
		_size = 0;
		while (lo < hi)
			_elem[_size++] = A[lo++];
	}
	void expand()
	{
		if (_size < _capacity)return;
		_capacity = (_capacity > DEFAULT_CAPACITY ? _capacity : DEFAULT_CAPACITY);
		T* old = _elem;
		_elem = new T[_capacity <<= 1];
		for (int i = 0; i < _size; i++)
			_elem[i] = old[i];
		delete[] old;
	}
	T& operator[](int r)
	{
		if (r >= 0 && r < _size)
			return _elem[r];
	}
	int insert(int r, T const& value)
	{
		expand();
		for (int i = _size; i > r; i--)
			_elem[i] = _elem[i - 1];
		_elem[r] = value;
		_size++;
		return r;
	}
	void remove(int lo, int hi)
	{
		if (lo >= hi)return;
		int delta = hi - lo;
		while (hi < _size)
			_elem[lo++] = _elem[hi++];
		_size -= delta;
	}
	~myVector() { if (_elem)delete[] _elem; }
};

struct Node {
	Node* prev;
	Node* next;
	myVector<char> balls;
	Node() : balls(DEFAULT_CAPACITY) { prev = nullptr; next = nullptr; }
	Node(int size) : balls(size) { prev = nullptr; next = nullptr; }
	int size() { return balls._size; }
} *HEAD, *END;


//寻址
Node* findRank(int r, int& cur_rank)
{
	Node* p = HEAD->next;	int count = 0;

	while (count <= r && p != END) {
		if (r < count + p->size()) {
			break;
		}
		count = count + p->size();
		p = p->next;
	}

	if (p == END)			//尾部插入，直接加在末节点尾端
	{
		p = p->prev;
		cur_rank = p->size();
	}
	else				//找到所在节点，以及元素在该节点中的rank
		cur_rank = r - count;
	return p;
}

//向左找到第一个value不同的位置及所在节点
void checkLeft(Node* p, Node*& lParent, int& head, int& count, const char& value)
{
	while (p != HEAD)
	{
		while (head >= 0 && p->balls[head] == value)
		{
			count++;	head--;
		}
		if (head < 0)		//该节点整个数组满足条件，继续前探
		{
			p = p->prev;
			head = p->size() - 1;
		}
		else				//该节点为左侧父节点
		{
			lParent = p;
			return;
		}
	}
	lParent = HEAD;
}

//向右找到第一个value的位置及所在节点
void checkRight(Node* p, Node*& rParent, int& tail, int& count, const char& value)
{
	while (p != END)
	{
		while (tail < p->size() && p->balls[tail] == value)
		{
			count++;	tail++;
		}
		if (tail >= p->size())
		{
			p = p->next;
			tail = 0;
		}
		else
		{
			rParent = p;
			return;
		}
	}
	rParent = END;
}

int main()
{
	char* a = new char[500005];
	cin.getline(a, 500005, '\n');

	int blockSize = sqrt(strlen(a));
	Node* nodes = new Node[10000];

	//初始化分块数组
	int top = 0;					//新节点在nodes中的编号
	HEAD = new Node(1);	END = new Node(1);
	HEAD->next = END;	END->prev = HEAD;
	Node* last = HEAD;				//依次向后衔接
	for (int i = 0; i < strlen(a); i += blockSize)
	{
		Node* tmp = &nodes[top++];
		if (strlen(a) - i < blockSize)	//最后一个节点，大小为余数
			tmp->balls.copyFrom(a, i, strlen(a));
		else
			tmp->balls.copyFrom(a, i, i + blockSize);
		tmp->prev = last;
		last->next = tmp;
		last = tmp;
	}
	last->next = END;
	END->prev = last;

	//读入记录
	int index, head, tail, m, count, cur_rank = 0;	char value;	Node* who, * lParent, * rParent;
	scanf("%d", &m);
	while (m--)
	{
		scanf("%d%*c%c", &index, &value);
		if (HEAD->next == END)
		{
			Node* tmp = &nodes[top++];
			HEAD->next = tmp;
			tmp->prev = HEAD;
			tmp->next = END;
			END->prev = tmp;
			tmp->balls.insert(index, value);
			continue;
		}

		who = findRank(index, cur_rank);
		who->balls.insert(cur_rank, value);
		head = cur_rank - 1;	tail = cur_rank;	count = 0;
		checkLeft(who, lParent, head, count, value);	//检查后，保证0<=head<=lParent->size()-1
		checkRight(who, rParent, tail, count, value);	//检查后，保证0<=tail<=rParent->size()-1

		//连环相消
		if (count <= 2) continue;

		Node* start = lParent, * end = rParent;
		char left = lParent->balls[head];
		char right = rParent->balls[tail];
		//寻找相消的开头、结束节点
		while (left == right)
		{
			lParent = start;	rParent = end;	count = 0;
			int iniHead = head, iniTail = tail;
			checkLeft(lParent, start, head, count, left);
			checkRight(rParent, end, tail, count, right);
			if (count < 3)	//连环结束,将头尾指针放回
			{
				head = iniHead;	tail = iniTail;
				start = lParent;	end = rParent;
				break;
			}
			left = start->balls[head];
			right = end->balls[tail];
		}

		//找到 [head,tail)，即将head向右挪动一格
		head = head + 1;
		while (head >= start->size())		//确保head不会走到 END:前提已经满足必定有相消位
		{
			start = start->next;
			head = 0;
		}

		if (start == end)					//位于同一节点，直接移除
			start->balls.remove(head, tail);
		else
		{									//位于不同节点，将中间节点从链表中移除
			start->balls.remove(head, start->size());
			end->balls.remove(0, tail);
			start->next = end;
			end->prev = start;
		}
	}

	Node* p = HEAD->next;	bool empty = true;
	while (p != END)
	{
		for (int i = 0; i < p->size(); i++)
			printf("%c", p->balls[i]);
		p = p->next;
	}
	printf("\n");
	delete[] nodes;
	delete[] a;
	return 0;
}
