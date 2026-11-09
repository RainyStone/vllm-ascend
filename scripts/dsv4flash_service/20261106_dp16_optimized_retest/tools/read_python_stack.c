#define Py_BUILD_CORE
#include <Python.h>
#include "internal/pycore_runtime.h"
#include "internal/pycore_frame.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
static int fd;
static int rd(uintptr_t a,void *p,size_t n){return pread(fd,p,n,(off_t)a)==n;}
static void text_at(PyObject *p,char *out,size_t cap){
 PyASCIIObject u;out[0]=0;if(!p||!rd((uintptr_t)p,&u,sizeof(u)))return;
 if(!u.state.compact||!u.state.ascii)return;
 size_t n=u.length<cap-1?(size_t)u.length:cap-1;
 if(rd((uintptr_t)p+sizeof(u),out,n))out[n]=0;
}
int main(int argc,char **argv){
 if(argc!=3)return 2;char path[80];snprintf(path,sizeof(path),"/proc/%s/mem",argv[1]);
 fd=open(path,O_RDONLY);if(fd<0){perror(path);return 3;}
 uintptr_t runtime=strtoull(argv[2],NULL,16);PyInterpreterState *interp;
 if(!rd(runtime+offsetof(_PyRuntimeState,interpreters.head),&interp,sizeof(interp)))return 4;
 PyInterpreterState it;if(!rd((uintptr_t)interp,&it,sizeof(it)))return 5;
 PyThreadState *thread=it.threads.head;
 for(int nt=0;thread&&nt<1000;nt++){
  PyThreadState ts;if(!rd((uintptr_t)thread,&ts,sizeof(ts)))break;
  printf("THREAD native=%lu python=%lu\n",ts.native_thread_id,ts.thread_id);
  _PyCFrame cf;
  if(ts.cframe&&rd((uintptr_t)ts.cframe,&cf,sizeof(cf))){
   _PyInterpreterFrame *frame=cf.current_frame;
   for(int nf=0;frame&&nf<80;nf++){
    _PyInterpreterFrame fr;if(!rd((uintptr_t)frame,&fr,sizeof(fr)))break;
    PyCodeObject co;if(!fr.f_code||!rd((uintptr_t)fr.f_code,&co,sizeof(co)))break;
    char filename[1024],name[256];text_at(co.co_filename,filename,sizeof(filename));text_at(co.co_name,name,sizeof(name));
    printf("  %s (%s:firstline=%d byteoffset=%lld)\n",name,filename,co.co_firstlineno,(long long)((uintptr_t)fr.prev_instr-((uintptr_t)fr.f_code+offsetof(PyCodeObject,co_code_adaptive))));
    frame=fr.previous;
   }
  }
  thread=ts.next;
 }
 close(fd);return 0;
}
