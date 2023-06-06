#include <fstream>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <string>

int gen(int num, char fname[20])
{
    FILE *fp = fopen(fname, "w+");

    fprintf(fp, "%d \n", num);

    srand(time(0));
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int x = 0;
            x = rand() % 1999 + (-999);
            fprintf(fp, "%d ", x);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}
int main()
{
    gen(10000, "10000.txt");
    gen(50000, "50000.txt");
    gen(100000, "100000.txt");
    gen(500000, "500000.txt");
    gen(1000000, "1000000.txt");
}