//#include<cuda_runtime.h>
//#include<duda_device_runtime_api.h>
//#ifndef __JETBRAINS_IDE__
//#include<cuda_fake_headers.h>
//#endif

#define HAVE_STRUCT_TIMESPEC
#include<pthread.h>
#include<cstdlib>
//#include<hash_map>
#include<cstring>
#include<map>
#include<iostream>


#define THREAD_NUM 8;
const double color_radius = 6.5;
pthread_rwlock_t* ralist_lock;

using std::cin;
using std::cout;
using std::endl;
using std::map;

namespace CuMeanShift{

/*	
#######################################################
##类名:				RAList
##功能描述：	
##数据成员描述:  	

#######################################################
*/
	class RAList{
		private:
			int exists;
			RAList* cur;
		public:
			int label;
			RAList* next;

		public:
			RAList(void);
			~RAList(void);
			int Insert(RAList *entry);
			
	};
	
	
		RAList::RAList( void )
		{
			label			= -1;
			next			= 0;	//NULL
		}
	
		RAList::~RAList( void )
		{}
	
		int RAList::Insert(RAList *entry)
		{
			if(!next)
			{
				next		= entry;
				entry->next = 0;
				return 0;
			}
			if(next->label > entry->label)
			{
				entry->next	= next;
				next		= entry;
				return 0;
			}
			exists	= 0;
			cur		= next;
			while(cur)
			{
				if(entry->label == cur->label)
				{
					exists = 1;
					break;
				}
				else if((!(cur->next))||(cur->next->label > entry->label))
				{
					entry->next	= cur->next;
					cur->next	= entry;
					break;
				}
				cur = cur->next;
			}
			return (int)(exists);
		}

/*	
#######################################################
##类名:				gaussCudaVer
##功能描述：		实现高斯函数计算、存储、输出工作 
##数据成员描述:  	
##	size：s值  
##	coefficient：	用于高斯函数计算 e的幂次的常数系数 函数值的常数系数 
##	result			存储高斯函数结果 
#######################################################
*/
	struct PthreadData
	{
		RAList *raNode1, *raNode2;
		RAList *raList1,*raList2;
		PthreadData(RAList* Node1, RAList* Node2,RAList* list1,RAList* list2) {
			raNode1 = Node1;
			raNode2 = Node2;
			raList1 = list1;
			raList2 = list2;
		}
	};


	void* thread_insert(void *arg) {
		pthread_rwlock_wrlock(&ralist_lock[((PthreadData*)arg)->raNode1->label]);//获取写入锁
		int insertRes=((PthreadData*)arg)->raList1->Insert(((PthreadData*)arg)->raNode2);
		pthread_rwlock_unlock(&ralist_lock[((PthreadData*)arg)->raNode1->label]);//释放写入锁
		if (insertRes != 0) {
			pthread_rwlock_wrlock(&ralist_lock[((PthreadData*)arg)->raNode2->label]);//获取写入锁
			((PthreadData*)arg)->raList2->Insert(((PthreadData*)arg)->raNode1);
			pthread_rwlock_unlock(&ralist_lock[((PthreadData*)arg)->raNode2->label]);//释放写入锁
		}
	}
/*	
#######################################################
##类名:				PthreadUnionFind
##功能描述：		用pthread 实现union_find
##数据成员描述:  	

#######################################################
*/
	
//	template <int blk_width=32,int blk_height=32,int chanels=1>

	class PthreadUnionFind{
		private:
			int width,height,pitch,regionCount;
			int *labels;
			int *new_labels;
			int *mode_point_count;
			float* mode;
			
		public:
			PthreadUnionFind(int *labels,
							float* mode,
							int* new_labels,
							int *mode_point_count,
							int pitch,
							int width,
							int height);
							
			void union_find_one_thread();
			void union_find_multi_thread();
			
		private:
			int dim_tran(int i,int j);
			void set_region_count();	
			float color_distance( const float* a, const float* b);					
			
	};
	
	PthreadUnionFind::PthreadUnionFind(int *labels,
							float* mode,
							int* new_labels,
							int *mode_point_count,
							int pitch,
							int width,
							int height){
		this->labels=labels;
		this->mode=mode;
		this->new_labels=new_labels;
		this->mode_point_count=mode_point_count;
		this->pitch=pitch;
		this->width=width;
		this->height=height;	
		set_region_count();					
		
	}
	
	void PthreadUnionFind::union_find_one_thread(){
		
		for(int counter = 0, deltaRegionCount = 1; counter<5 && deltaRegionCount>0; counter++)
		{
			int temp_count = 0;
			// 1.Build RAM using classifiction structure
			RAList *raList = new RAList [regionCount], *raPool = new RAList [10*regionCount];	//10 is hard coded!
			for(int i = 0; i < regionCount; i++)
			{
				raList[i].label = i;
				raList[i].next = NULL;
			}
			for(int i = 0; i < regionCount*10-1; i++)
			{
				raPool[i].next = &raPool[i+1];
			}
			raPool[10*regionCount-1].next = NULL;
			RAList	*raNode1, *raNode2, *oldRAFreeList, *freeRAList = raPool;
			for(int i=0;i<height;i++) 
				for(int j=0;j<width;j++)
				{
					if(i>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i-1,j)])
					{
						// Get 2 free node
						raNode1			= freeRAList;
						raNode2			= freeRAList->next;
						oldRAFreeList	= freeRAList;
						freeRAList		= freeRAList->next->next;
						temp_count += 2;
						// connect the two region
						raNode1->label	= labels[dim_tran(i,j)];
						raNode2->label	= labels[dim_tran(i-1,j)];
						if (raList[labels[dim_tran(i, j)]].Insert(raNode2))	//already exists!
						{
							freeRAList = oldRAFreeList;
							temp_count -= 2;
						}
						else
							raList[labels[dim_tran(i-1,j)]].Insert(raNode1);
					}
					if(j>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i,j-1)])
					{
						// Get 2 free node
						raNode1			= freeRAList;
						raNode2			= freeRAList->next;
						oldRAFreeList	= freeRAList;
						freeRAList		= freeRAList->next->next;
						temp_count += 2;
						// connect the two region
						raNode1->label	= labels[dim_tran(i,j)];
						raNode2->label	= labels[dim_tran(i,j-1)];
						if (raList[labels[dim_tran(i, j)]].Insert(raNode2))
						{
							freeRAList = oldRAFreeList;
							temp_count -= 2;
						}
						else
							raList[labels[dim_tran(i,j-1)]].Insert(raNode1);
					}
				}

				// 2.Treat each region Ri as a disjoint set
				for(int i = 0; i < regionCount; i++)
				{
					RAList	*neighbor = raList[i].next;
					while(neighbor)
					{
						if(color_distance(&mode[3*i], &mode[3*neighbor->label])<color_radius*color_radius)
						{
							int iCanEl = i, neighCanEl	= neighbor->label;
							while(raList[iCanEl].label != iCanEl) iCanEl = raList[iCanEl].label;
							while(raList[neighCanEl].label != neighCanEl) neighCanEl = raList[neighCanEl].label;
							if(iCanEl<neighCanEl)
								raList[neighCanEl].label = iCanEl;
							else
							{
								//raList[raList[iCanEl].label].label = iCanEl;
								raList[iCanEl].label = neighCanEl;
							}
						}
						neighbor = neighbor->next;
					}
				}
				// 3. Union Find
				for(int i = 0; i < regionCount; i++)
				{
					int iCanEl	= i;
					while(raList[iCanEl].label != iCanEl) iCanEl	= raList[iCanEl].label;
					raList[i].label	= iCanEl;
				}
				// 4. Traverse joint sets, relabeling image.
				int *mode_point_count_buffer = new int[regionCount];
				memset(mode_point_count_buffer, 0, regionCount*sizeof(int));
				float *mode_buffer = new float[regionCount*3];
				int	*label_buffer = new int[regionCount];

				for(int i=0;i<regionCount; i++)
				{
					label_buffer[i]	= -1;
					mode_buffer[i*3+0] = 0;
					mode_buffer[i*3+1] = 0;
					mode_buffer[i*3+2] = 0;
				}
				for(int i=0;i<regionCount; i++)
				{
					int iCanEl	= raList[i].label;
					mode_point_count_buffer[iCanEl] += mode_point_count[i];
					for(int k=0;k<3;k++)
						mode_buffer[iCanEl*3+k] += mode[i*3+k]*mode_point_count[i];
				}
				int	label = -1;
				for(int i = 0; i < regionCount; i++)
				{
					int iCanEl	= raList[i].label;
					if(label_buffer[iCanEl] < 0)
					{
						label_buffer[iCanEl]	= ++label;

						for(int k = 0; k < 3; k++)
							mode[label*3+k]	= (mode_buffer[iCanEl*3+k])/(mode_point_count_buffer[iCanEl]);

						mode_point_count[label]	= mode_point_count_buffer[iCanEl];
					}
				}
				regionCount = label+1;
				for(int i = 0; i < height; i++)
					for(int j = 0; j < width; j++)
						labels[dim_tran(i,j)]	= label_buffer[raList[labels[dim_tran(i,j)]].label];

				delete [] mode_buffer;
				delete [] mode_point_count_buffer;
				delete [] label_buffer;

				//Destroy RAM
				delete[] raList;
				delete[] raPool;

//				deltaRegionCount = oldRegionCount - regionCount;
//				oldRegionCount = regionCount;
				cout<<"Mean Shift(TransitiveClosure):"<<regionCount<<endl;
		}
		
	}

	void PthreadUnionFind::union_find_multi_thread()
	{
		for(int counter = 0, deltaRegionCount = 1; counter<5 && deltaRegionCount>0; counter++)
		{
			int temp_count = 0;
			// 1.Build RAM using classifiction structure
			ralist_lock = new pthread_rwlock_t[regionCount];
			pthread_t *thread_pool=new pthread_t[regionCount];
			RAList *raList = new RAList[regionCount];


			for(int i = 0; i < regionCount; i++)
			{
				raList[i].label = i;
				raList[i].next = NULL;
			}
			RAList	*raNode1, *raNode2;
			for(int i=0;i<height;i++) 
				for(int j=0;j<width;j++)
				{
					if(i>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i-1,j)])
					{
						// Get 2 free node
						raNode1	= new RAList();
						raNode2 = new RAList();
						// connect the two region
						raNode1->label	= labels[dim_tran(i,j)];
						raNode2->label	= labels[dim_tran(i-1,j)];
						PthreadData data(raNode1, raNode2, &raList[raNode1->label], &raList[raNode2->label]);
						/*
						 to be continued
						*/

						
						
					}
					if(j>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i,j-1)])
					{
						// Get 2 free node
						raNode1 = new RAList();
						raNode2	= new RAList();
						// connect the two region
						raNode1->label	= labels[dim_tran(i,j)];
						raNode2->label	= labels[dim_tran(i,j-1)];
						if (raList[labels[dim_tran(i, j)]].Insert(raNode2))
						{
						}
						else
							raList[labels[dim_tran(i,j-1)]].Insert(raNode1);
					}
				}

				// 2.Treat each region Ri as a disjoint set
				for(int i = 0; i < regionCount; i++)
				{
					RAList	*neighbor = raList[i].next;
					while(neighbor)
					{
						if(color_distance(&mode[3*i], &mode[3*neighbor->label])<color_radius*color_radius)
						{
							int iCanEl = i, neighCanEl	= neighbor->label;
							while(raList[iCanEl].label != iCanEl) iCanEl = raList[iCanEl].label;
							while(raList[neighCanEl].label != neighCanEl) neighCanEl = raList[neighCanEl].label;
							if(iCanEl<neighCanEl)
								raList[neighCanEl].label = iCanEl;
							else
							{
								//raList[raList[iCanEl].label].label = iCanEl;
								raList[iCanEl].label = neighCanEl;
							}
						}
						neighbor = neighbor->next;
					}
				}
				// 3. Union Find
				for(int i = 0; i < regionCount; i++)
				{
					int iCanEl	= i;
					while(raList[iCanEl].label != iCanEl) iCanEl	= raList[iCanEl].label;
					raList[i].label	= iCanEl;
				}
				// 4. Traverse joint sets, relabeling image.
				int *mode_point_count_buffer = new int[regionCount];
				memset(mode_point_count_buffer, 0, regionCount*sizeof(int));
				float *mode_buffer = new float[regionCount*3];
				int	*label_buffer = new int[regionCount];

				for(int i=0;i<regionCount; i++)
				{
					label_buffer[i]	= -1;
					mode_buffer[i*3+0] = 0;
					mode_buffer[i*3+1] = 0;
					mode_buffer[i*3+2] = 0;
				}
				for(int i=0;i<regionCount; i++)
				{
					int iCanEl	= raList[i].label;
					mode_point_count_buffer[iCanEl] += mode_point_count[i];
					for(int k=0;k<3;k++)
						mode_buffer[iCanEl*3+k] += mode[i*3+k]*mode_point_count[i];
				}
				int	label = -1;
				for(int i = 0; i < regionCount; i++)
				{
					int iCanEl	= raList[i].label;
					if(label_buffer[iCanEl] < 0)
					{
						label_buffer[iCanEl]	= ++label;

						for(int k = 0; k < 3; k++)
							mode[label*3+k]	= (mode_buffer[iCanEl*3+k])/(mode_point_count_buffer[iCanEl]);

						mode_point_count[label]	= mode_point_count_buffer[iCanEl];
					}
				}
				regionCount = label+1;
				for(int i = 0; i < height; i++)
					for(int j = 0; j < width; j++)
						labels[dim_tran(i,j)]	= label_buffer[raList[labels[dim_tran(i,j)]].label];

				delete [] mode_buffer;
				delete [] mode_point_count_buffer;
				delete [] label_buffer;

				//Destroy RAM
				delete[] raList;

//				deltaRegionCount = oldRegionCount - regionCount;
//				oldRegionCount = regionCount;
				cout<<"Mean Shift(TransitiveClosure):"<<regionCount<<endl;
		}
	}
	
	
	int PthreadUnionFind::dim_tran(int i, int j){
		return i*width+j;
	}
	
	void PthreadUnionFind::set_region_count(){
//		hash_map<int,bool> hmap;
		map<int,bool> hmap;
		for (int i=0; i<height; i++){
			for (int j=0;j<width; j++){
				if(hmap.find(labels[dim_tran(i,j)])==hmap.end()){
					hmap[labels[dim_tran(i,j)]]=true;
				} 
			}
		}	
		regionCount=hmap.size();
//		cout<<"regionCount:"<<regionCount<<endl;
		return;
	} 
	float PthreadUnionFind::color_distance( const float* a, const float* b){
 		float l = a[0]-b[0], u=a[1]-b[1], v=a[2]-b[2];
		return l*l+u*u+v*v;
	}
}


void test_union_find(){
	int width,height,pitch,label_count;
	int *labels;
	int *new_labels;
	int *mode_point_count;
	float* mode;	

	cout<<"input height,width:";
	cin>>height>>width;
	cout<<"input label_count:";
	cin>>label_count;

	labels=new int[height*width];
	new_labels=new int[height*width];
	mode=new float[height*width*3];
	mode_point_count = new int[height*width];
	memset(mode_point_count, 0, width*height * sizeof(int));
	for (int i=0; i<height; i++){
		for (int j=0; j<width;j++){
			labels[i*width+j]=rand()%label_count;
			mode_point_count[labels[i*width + j]] += 1;
			new_labels[i*width+j]=0;
			mode[i*width+j]=float(rand()%256);
			mode[i*width+j+1]=float(rand()%256);
			mode[i*width+j+2]=float(rand()%256);
		}
	}
	cout<<"start new puf\n";
	CuMeanShift::PthreadUnionFind puf(labels,mode,new_labels,mode_point_count,100,width,height);
	cout<<"end new puf\n";
	puf.union_find_one_thread();
}

int main(int argc, char const *argv[])
{
	test_union_find();
	return 0;
}


