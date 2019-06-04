#ifdef __JETBRAINS_IDE__

int cuda_main(int argc, char *argv[]);
int main(int argc, char *argv[])
{
    return cuda_main(argc, argv);
}
#endif