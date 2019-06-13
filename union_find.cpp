


#define HAVE_STRUCT_TIMESPEC
#include <CImg.h>
#include <cmath>
#include <stack>
#include <fstream>
#include<pthread.h>
#include<cstdlib>
#include<cstring>
#include<map>
#include<iostream>
#include<time.h>


#define THREAD_NUM 8
const double color_radius = 6.5;


using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::ofstream;
using namespace cimg_library;
using namespace std;

struct Point {
    int x, y;

    Point(int x, int y) : x(x), y(y) {}
};
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
    int insert(RAList *entry);
    RAList* merge(RAList *head1, RAList *head2);



};


RAList::RAList( void )
{
    label			= -1;
    next			= 0;	//NULL
}

RAList::~RAList( void )
{}

int RAList::insert(RAList *entry)
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



RAList *RAList::merge(RAList *head1, RAList *head2) {
    if (head1 == NULL)return head2;
    if (head2 == NULL)return head1;
    RAList *res, *p;
    if (head1->label < head2->label) {
        res = head1;
        head1 = head1->next;
    } else if(head1->label>head2->label){
        res = head2;
        head2 = head2->next;
    }
    else{
        res=head1;
        head1=head1->next;
        head2=head2->next;
    }
    p = res;

    while (head1 != NULL && head2 != NULL) {
        if (head1->label < head2->label) {
            p->next = head1;
            head1 = head1->next;
        } else if (head1->label>head2->label){
            p->next = head2;
            head2 = head2->next;
        }
        else{
            p->next=head2;
            head1=head1->next;
            head2=head2->next;
        }
        p = p->next;
    }
    if (head1 != NULL)p->next = head1;
    else if (head2 != NULL)p->next = head2;
    return res;
}


struct PthreadData
{

    int start, end,width,height;
    int* labels;
    RAList* ralist;

    PthreadData(int i_start=0, int i_end=0, int i_width=0,int i_height=0,int* i_labels=NULL, RAList* i_ralist=NULL) {
        start = i_start;
        end = i_end;
        width = i_width;
        height=i_height;
        labels = i_labels;
        ralist = i_ralist;
    }
    PthreadData(PthreadData* new_data) {
        start = new_data->start;
        end = new_data->end;
        width = new_data->width;
        height=new_data->height;
        labels = new_data->labels;
        ralist = new_data -> ralist;
    }
};


void *threadInsert(void *arg) {
    PthreadData* data = (PthreadData*)arg;

    for (int i =data->start; i <= data->end; i++) {
        if (i >= data->width&&i<data->width*data->height&&data->labels[i] != data->labels[i - data->width]) {
            RAList* ra_node1 = new RAList();
            RAList* ra_node2 = new RAList();
            ra_node1->label = data->labels[i];
            ra_node2->label = data->labels[i - data->width];
//			pthread_rwlock_wrlock(&ralist_lock[ra_node1->label]);
            int insert_res=(data->ralist)[ra_node1->label].insert(ra_node2);

//			pthread_rwlock_unlock(&ralist_lock[ra_node1->label]);
            if (insert_res != 0) {
//				pthread_rwlock_wrlock(&ralist_lock[ra_node2->label]);
                (data->ralist)[ra_node2->label].insert(ra_node1);
//				pthread_rwlock_unlock(&ralist_lock[ra_node2->label]);
            }
        }
        if (i%data->width!=0&&i<data->width*data->height&&data->labels[i] != data->labels[i - 1]) {
            RAList* ra_node1 = new RAList();
            RAList* ra_node2 = new RAList();
            ra_node1->label = data->labels[i];
            ra_node2->label = data->labels[i - 1];
//			pthread_rwlock_wrlock(&ralist_lock[ra_node1->label]);
            int insert_res=(data->ralist)[ra_node1->label].insert(ra_node2);
//			pthread_rwlock_unlock(&ralist_lock[ra_node1->label]);
            if (insert_res != 0) {
//				pthread_rwlock_wrlock(&ralist_lock[ra_node2->label]);
                (data->ralist)[ra_node2->label].insert(ra_node1);
//				pthread_rwlock_unlock(&ralist_lock[ra_node2->label]);
            }
        }
    }
    pthread_exit(0);
    return NULL;
}




class PthreadUnionFind{
private:
    int width,height,region_count;
    int *labels;
    int *new_labels;
    int *mode_point_count;
    float* mode;

public:
    PthreadUnionFind(int *labels,
                     float* mode,
                     int* new_labels,
                     int *mode_point_count,

                     int width,
                     int height);
    PthreadUnionFind(CImg<uint8_t> &img, CImg<int> &labels);

    void unionFindOneThread();
    void unionFindMultiThread();
    int* get_result_labels();

private:
    int dim_tran(int i,int j);
    void set_region_count();
    float color_distance( const float* a, const float* b);

};

PthreadUnionFind::PthreadUnionFind(int *labels,
                                   float* mode,
                                   int* new_labels,
                                   int *mode_point_count,
                                   int width,
                                   int height){
    this->labels=labels;
    this->mode=mode;
    this->new_labels=new_labels;
    this->mode_point_count=mode_point_count;
    this->width=width;
    this->height=height;
    set_region_count();

}
PthreadUnionFind::PthreadUnionFind(CImg<uint8_t> &img, CImg<int> &cimg_labels) {
    region_count = 0;
    width = img.width();
    height = img.height();
    labels = new int[width*height];
    new_labels = new int[width*height];
    mode_point_count = new int[height*width];
    mode = new float[img.height() * img.width() * 3];

    memset(mode_point_count, 0, width*height * sizeof(int));
    memset(new_labels, 0, width*height * sizeof(int));

    int label = -1;
    for (int i = 0; i < img.height(); i++) {
        for (int j = 0; j < img.width(); j++) {
            labels[dim_tran(i,j)] = ++label;
            mode_point_count[label] += 1;
            mode[label * 3 + 0] = img(j, i, 0);
            mode[label * 3 + 1] = img(j, i, 1);
            mode[label * 3 + 2] = img(j, i, 2);

            // Fill
            std::stack<Point> neigh_stack;
            neigh_stack.push(Point(i, j));
            const int dxdy[][2] = {{-1, -1},
                                   {-1, 0},
                                   {-1, 1},
                                   {0,  -1},
                                   {0,  1},
                                   {1,  -1},
                                   {1,  0},
                                   {1,  1}};
            while (!neigh_stack.empty()) {
                Point p = neigh_stack.top();
                neigh_stack.pop();
                for (int k = 0; k < 8; k++) {
                    int i2 = p.x + dxdy[k][0], j2 = p.y + dxdy[k][1];
                    if (i2 >= 0 && j2 >= 0 && i2 < img.height() && j2 < img.width() && labels[dim_tran(i2, j2)] < 0 &&
                        img(j,i) == img(j2, i2)) {
                        labels[dim_tran(i2, j2)] = label;
                        mode_point_count[label] += 1;
                        neigh_stack.push(Point(i2, j2));
                    }
                }
            }

        }
    }
    //current Region count
    region_count = label + 1;
}

void PthreadUnionFind::unionFindOneThread(){
    int old_region_count=region_count;
    for(int counter = 0, delta_region_count = 1; counter<5 && delta_region_count>0; counter++)
    {
        int temp_count = 0;
        // 1.Build RAM using classifiction structure
        RAList *ralist = new RAList [region_count], *raPool = new RAList [10*region_count];	//10 is hard coded!
        for(int i = 0; i < region_count; i++)
        {
            ralist[i].label = i;
            ralist[i].next = NULL;
        }
        for(int i = 0; i < region_count*10-1; i++)
        {
            raPool[i].next = &raPool[i+1];
        }
        raPool[10*region_count-1].next = NULL;
        RAList	*ra_node1, *ra_node2, *old_ra_freelist, *free_ra_list = raPool;
        for(int i=0;i<height;i++)
            for(int j=0;j<width;j++)
            {
                if(i>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i-1,j)])
                {
                    // Get 2 free node
                    ra_node1			= free_ra_list;
                    ra_node2			= free_ra_list->next;
                    old_ra_freelist	= free_ra_list;
                    free_ra_list		= free_ra_list->next->next;
                    temp_count += 2;
                    // connect the two region
                    ra_node1->label	= labels[dim_tran(i,j)];
                    ra_node2->label	= labels[dim_tran(i-1,j)];
                    if (ralist[labels[dim_tran(i, j)]].insert(ra_node2))	//already exists!
                    {
                        free_ra_list = old_ra_freelist;
                        temp_count -= 2;
                    }
                    else
                        ralist[labels[dim_tran(i-1,j)]].insert(ra_node1);
                }
                if(j>0 && labels[dim_tran(i,j)]!=labels[dim_tran(i,j-1)])
                {
                    // Get 2 free node
                    ra_node1			= free_ra_list;
                    ra_node2			= free_ra_list->next;
                    old_ra_freelist	= free_ra_list;
                    free_ra_list		= free_ra_list->next->next;
                    temp_count += 2;
                    // connect the two region
                    ra_node1->label	= labels[dim_tran(i,j)];
                    ra_node2->label	= labels[dim_tran(i,j-1)];
                    if (ralist[labels[dim_tran(i, j)]].insert(ra_node2))
                    {
                        free_ra_list = old_ra_freelist;
                        temp_count -= 2;
                    }
                    else
                        ralist[labels[dim_tran(i,j-1)]].insert(ra_node1);
                }
            }

        // 2.Treat each region Ri as a disjoint set
        for(int i = 0; i < region_count; i++)
        {
            RAList	*neighbor = ralist[i].next;
            while(neighbor)
            {
                if(color_distance(&mode[3*i], &mode[3*neighbor->label])<color_radius*color_radius)
                {
                    int i_can_el = i, neigh_can_el	= neighbor->label;
                    while(ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                    while(ralist[neigh_can_el].label != neigh_can_el) neigh_can_el = ralist[neigh_can_el].label;
                    if(i_can_el<neigh_can_el)
                        ralist[neigh_can_el].label = i_can_el;
                    else
                    {
                        //ralist[ralist[i_can_el].label].label = i_can_el;
                        ralist[i_can_el].label = neigh_can_el;
                    }
                }
                neighbor = neighbor->next;
            }
        }
        // 3. Union Find
        for(int i = 0; i < region_count; i++)
        {
            int i_can_el	= i;
            while(ralist[i_can_el].label != i_can_el) i_can_el	= ralist[i_can_el].label;
            ralist[i].label	= i_can_el;
        }
        // 4. Traverse joint sets, relabeling image.
        int *mode_point_count_buffer = new int[region_count];
        memset(mode_point_count_buffer, 0, region_count*sizeof(int));
        float *mode_buffer = new float[region_count*3];
        int	*label_buffer = new int[region_count];

        for(int i=0;i<region_count; i++)
        {
            label_buffer[i]	= -1;
            mode_buffer[i*3+0] = 0;
            mode_buffer[i*3+1] = 0;
            mode_buffer[i*3+2] = 0;
        }
        for(int i=0;i<region_count; i++)
        {
            int i_can_el	= ralist[i].label;
            mode_point_count_buffer[i_can_el] += mode_point_count[i];
            for(int k=0;k<3;k++)
                mode_buffer[i_can_el*3+k] += mode[i*3+k]*mode_point_count[i];
        }
        int	label = -1;
        for(int i = 0; i < region_count; i++)
        {
            int i_can_el	= ralist[i].label;
            if(label_buffer[i_can_el] < 0)
            {
                label_buffer[i_can_el]	= ++label;

                for(int k = 0; k < 3; k++)
                    mode[label*3+k]	= (mode_buffer[i_can_el*3+k])/(mode_point_count_buffer[i_can_el]);

                mode_point_count[label]	= mode_point_count_buffer[i_can_el];
            }
        }
        region_count = label+1;
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
                labels[dim_tran(i,j)]	= label_buffer[ralist[labels[dim_tran(i,j)]].label];

        delete [] mode_buffer;
        delete [] mode_point_count_buffer;
        delete [] label_buffer;

        //Destroy RAM
        delete[] ralist;
        delete[] raPool;

        delta_region_count = old_region_count - region_count;
        old_region_count= region_count;
        cout<<"Mean Shift(TransitiveClosure):"<<region_count<<endl;
    }

}

void PthreadUnionFind::unionFindMultiThread()
{
//		ralist_lock = new pthread_rwlock_t[region_count];
//		pthread_t *thread_pool=new pthread_t[THREAD_NUM];
//		PthreadData *pthread_data_array = new PthreadData[THREAD_NUM];
    for(int counter = 0, delta_region_count = 1; counter<5 && delta_region_count>0; counter++)
    {
        pthread_t *thread_pool=new pthread_t[THREAD_NUM];
        PthreadData *pthread_data_array = new PthreadData[THREAD_NUM];
        RAList *ralist = new RAList[region_count];
        RAList *ralist_array=new RAList[THREAD_NUM*region_count];
        int block_size = width*height / THREAD_NUM;
        void * thread_status;
        int thread_res;
        int old_region_count=region_count;
        for(int i = 0; i < region_count; i++)
        {
            ralist[i].label = i;
            ralist[i].next = NULL;
        }
        for (int i=0; i<THREAD_NUM*region_count; i++){
            ralist_array[i].label=i%region_count;
            ralist_array[i].next=NULL;
        }
        for (int i = 0; i < THREAD_NUM; i++) {
            int end = (i + 1)*block_size > width*height ? (i + 1)*block_size - 1 : width*width - 1;
            pthread_data_array[i] = new PthreadData(i*block_size, end,width, height,labels, &ralist_array[i*region_count]);
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            thread_res=pthread_create(&thread_pool[i], NULL, threadInsert, &pthread_data_array[i]);
            if (thread_res != 0)
            {
                perror("Thread creation failed");
                exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < THREAD_NUM; i++) {
            thread_res=pthread_join(thread_pool[i], &thread_status);
            if (thread_res != 0)
            {
                perror("Thread creation failed");
                exit(EXIT_FAILURE);
            }
        }
        for (int i=0; i<THREAD_NUM*region_count; i++){
            ralist[i%region_count]=*(ralist[i%region_count].merge(&ralist[i%region_count],&ralist_array[i]));
        }

        // 2.Treat each region Ri as a disjoint set
        for(int i = 0; i < region_count; i++)
        {
            RAList	*neighbor = ralist[i].next;
            while(neighbor)
            {
                if(color_distance(&mode[3*i], &mode[3*neighbor->label])<color_radius*color_radius)
                {
                    int i_can_el = i, neigh_can_el	= neighbor->label;
                    while(ralist[i_can_el].label != i_can_el) i_can_el = ralist[i_can_el].label;
                    while(ralist[neigh_can_el].label != neigh_can_el) neigh_can_el = ralist[neigh_can_el].label;
                    if(i_can_el<neigh_can_el)
                        ralist[neigh_can_el].label = i_can_el;
                    else
                    {
                        //ralist[ralist[i_can_el].label].label = i_can_el;
                        ralist[i_can_el].label = neigh_can_el;
                    }
                }
                neighbor = neighbor->next;
            }
        }
        // 3. Union Find
        for(int i = 0; i < region_count; i++)
        {
            int i_can_el	= i;
            while(ralist[i_can_el].label != i_can_el) i_can_el	= ralist[i_can_el].label;
            ralist[i].label	= i_can_el;
        }
        // 4. Traverse joint sets, relabeling image.
        int *mode_point_count_buffer = new int[region_count];
        memset(mode_point_count_buffer, 0, region_count*sizeof(int));
        float *mode_buffer = new float[region_count*3];
        int	*label_buffer = new int[region_count];

        for(int i=0;i<region_count; i++)
        {
            label_buffer[i]	= -1;
            mode_buffer[i*3+0] = 0;
            mode_buffer[i*3+1] = 0;
            mode_buffer[i*3+2] = 0;
        }
        for(int i=0;i<region_count; i++)
        {
            int i_can_el	= ralist[i].label;
            mode_point_count_buffer[i_can_el] += mode_point_count[i];
            for(int k=0;k<3;k++)
                mode_buffer[i_can_el*3+k] += mode[i*3+k]*mode_point_count[i];
        }
        int	label = -1;
        for(int i = 0; i < region_count; i++)
        {
            int i_can_el	= ralist[i].label;
            if(label_buffer[i_can_el] < 0)
            {
                label_buffer[i_can_el]	= ++label;

                for(int k = 0; k < 3; k++)
                    mode[label*3+k]	= (mode_buffer[i_can_el*3+k])/(mode_point_count_buffer[i_can_el]);

                mode_point_count[label]	= mode_point_count_buffer[i_can_el];
            }
        }
        region_count = label+1;
        for(int i = 0; i < height; i++)
            for(int j = 0; j < width; j++)
                labels[dim_tran(i,j)]	= label_buffer[ralist[labels[dim_tran(i,j)]].label];

        delete [] mode_buffer;
        delete [] mode_point_count_buffer;
        delete [] label_buffer;

        //Destroy RAM
        delete[] ralist;
        delete[] thread_pool;
        delete[] pthread_data_array;
        delete[] ralist_array;

        delta_region_count = old_region_count - region_count;
        old_region_count = region_count;
        cout<<"Mean Shift(TransitiveClosure):"<<region_count<<endl;
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
    region_count=hmap.size();
//		cout<<"region_count:"<<region_count<<endl;
    return;
}
float PthreadUnionFind::color_distance( const float* a, const float* b){
    float l = a[0]-b[0], u=a[1]-b[1], v=a[2]-b[2];
    return l*l+u*u+v*v;
}

int* PthreadUnionFind::get_result_labels(){
    return labels;
}





bool outputBin(const char *path, size_t count, int *data) {
    ofstream file(path, ios_base::out | ios_base::trunc | ios_base::binary);
    if (!file.is_open())
        return false;
    file.write((const char*)&count, sizeof(size_t));
    file.write((const char *) data, count * sizeof(int));
    file.close();
    return true;
}

/**
 * @brief: first argument: input image path,
 *         second argument: output binfile path
 */
int main(int argc, char **argv) {
    /// read image
    if (argc != 3) {
        cout << "Invalid argument number: " << argc - 1 << ", required is 2" << endl;
        return 1;
    }
    CImg <uint8_t> img(argv[1]), org_img(img);
    CImg<int> labels(img.width(), img.height(), 1, 1, -1);
    if (img.is_empty()) {
        cout << "Failed to read image" << endl;
        return 1;
    } else if (img.spectrum() != 3) {
        cout << "Image should be 3-channels, get " << img.spectrum() << " channels" << endl;
        return 1;
    }


//        CImgDisplay disp(img, "img", 1);
//        disp.show();
//        while (!disp.is_closed()) {
//            disp.wait();
//        }

    clock_t start,finish;
    double totaltime;

    start=clock();
    PthreadUnionFind puf1(img, labels);
    puf1.unionFindOneThread();
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"\n one_thread runtime of file "<<argv[1]<<" is "<<totaltime<<"second"<<endl;

    start=clock();
    PthreadUnionFind puf2(img, labels);
    puf2.unionFindMultiThread();
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"\n multi_thread runtime of file "<<argv[1]<<" is "<<totaltime<<"second"<<endl;
    outputBin(argv[2],labels.size(),puf2.get_result_labels());
}

