#include <iostream>
#include <stack>
#include <dirent.h>
#include <stdio.h>
#include <pthread.h>
#include <map>
#include <stdint.h>
#include <ctime>
#include <fstream>

using namespace std;
/*
Opens a directory and constructs a frequency map of all words in all files in that directory.
Multithreaded
*/

const int NUMTHREADS = 1250;
//thr   time    check   freqS
//1   = 9.70839 4794910 116349
//2   = 5.81199 4794910 116349
//5   = 2.54553 4794910 116349
//25  = .516524 4794910 116349
//125 = .104196 4794910 116349
//1250= .00994938 474910 116349
const char* path;

pthread_mutex_t mut_files;
stack<string> files;

pthread_mutex_t mut_freq;
map<string,uint64_t> global_freq;
uint64_t totalwords = 0;

//timing
double prevTime;
void startChrono() {
        prevTime = double(clock()) / CLOCKS_PER_SEC;
}
double stopChrono() {
        double currTime = double(clock()) / CLOCKS_PER_SEC;
        return currTime - prevTime;
}

void* parseFiles(void* vargs){
	map<string, uint64_t> local_freq;
	bool keep_working = true;

	while(keep_working){
		//lock mutex
		pthread_mutex_lock(&mut_files);
		//cout << files.size() << " in queue !" << endl;
		if (files.size() == 0){
			keep_working = false;
			pthread_mutex_unlock(&mut_files);
		}else{
			string filename = path + files.top();
			files.pop();
			//cout << "file : " << filename << endl;
			pthread_mutex_unlock(&mut_files);

			ifstream file;
			file.open(filename.c_str());
			if (!file){
				cerr << "file " << filename << " not open! make sure the provided path name has a trailing '/'" << endl;
			}else{
				string word;
				while (file >> word){
					//cout << word << endl;
					//word not in map? => init at zero
					if ( local_freq.find(word) == local_freq.end())
						local_freq[word] = 0;
					
					//++
					local_freq[word]++;
					totalwords++;
				}
				file.close();

				//add to global
				pthread_mutex_lock(&mut_freq);
				for (map<string,uint64_t>::iterator it=local_freq.begin(); it != local_freq.end() ; it++){
					//cout << "adding" << it->first << " - " << it->second << " to global" << endl;
					if (global_freq.find(it->first) == global_freq.end())
						global_freq[it->first] = it->second;
					else
						global_freq[it->first] += it->second;
				}
				pthread_mutex_unlock(&mut_freq);
				local_freq.clear();
			}
		}
	}

	//return
	return NULL;
}

int main(int argc, char* argv[]){
	if (argc == 2){
		path = argv[1];
	}else{
		path = "/home/drew/done/megaTest/"; //mind trailing / !!
	}

	pthread_mutex_init(&mut_files,NULL);
	pthread_mutex_init(&mut_freq,NULL);

	//get file list of start dir
	cout << "reading directory: " << path << " ..." << endl;
	pthread_mutex_lock(&mut_files);
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (path)) != NULL) {
		while ((ent = readdir (dir)) != NULL) {
			string name = ent->d_name;

			if ( !(name == "." || name == "..") ){
				files.push(name);
			}
		}
		closedir (dir);
	} else {
		cerr << path << " not found! exiting... " << endl;
		return -1;
	}
	pthread_mutex_unlock(&mut_files);
	cout << "found " << files.size() << " files" << endl;

	//threads
	startChrono();
	cout << "parsing using " << NUMTHREADS << " threads" << endl;
	pthread_t *thread = new pthread_t[NUMTHREADS];
	for (int i = 0 ; i < NUMTHREADS; i++)
		pthread_create(&thread[i],NULL,parseFiles,NULL);
	
	for (int i = 0; i < NUMTHREADS; i++)
		pthread_join(thread[i],NULL);

	uint64_t checkVal = 0;
	for (map<string,uint64_t>::iterator it = global_freq.begin() ; it != global_freq.end(); it++){
		cout << it->first << " - " << it->second / (1.0 * totalwords) << endl;
	}

	//approx of wall time = total time of each thread on each core / numthreads
	cout << "runtime: " << stopChrono() / NUMTHREADS << endl;
	
	delete [] thread;
	pthread_mutex_destroy(&mut_files);
	pthread_mutex_destroy(&mut_freq);

	return 0;
}