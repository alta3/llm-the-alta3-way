# **kubectl get**

<video class="slide" controls="" controlslist="nodownload" data-autoplay="" poster="https://labs.alta3.com/courses/kubernetes/slides/vid/poster.PNG"><source type="video/mp4" src="https://labs.alta3.com/courses/kubernetes/slides/05-Listing-Resources-with-kubectl-get.mp4" data-lazy-loaded=""></video>

### CKAD Objectives
- *Understand debugging in Kubernetes*

### Lab Objective
One of the first commands you learn, and one that may be of utmost importance down the line, is the *get* command. Although small and simple, you'll find that the *get* command is a useful tool for gathering information inside your environment. You will <a href="https://kubernetes.io/docs/reference/kubectl/cheatsheet/#viewing-finding-resources" target="_blank">discover</a> that you will be using this command quite a bit as soon as you start running deployments https://kubernetes.io/docs/reference/kubectl/cheatsheet/#viewing-finding-resources.

While issuing commands, beware of a bad habit that can form.  For example, at the CLI, you run the `get` command without the `kubectl` prefix. More than once you might wonder, "Why doesn't this work?" Remember that Kubernetes commands require a preface of *kubectl*, then a space, and then the command.

We will issue the kubectl `get` command to list several different elements within a Kubernetes environment.

The `get` command in kubectl command line interface offers a rich set of options to acquire more or less information. Use *kubectl get* or *kubectl get --help* for even more information.

Kubectl Cheatsheet:  
<a target="_blank" href="https://kubernetes.io/docs/reference/kubectl/cheatsheet/"> https://kubernetes.io/docs/reference/kubectl/cheatsheet/</a>


### Questions
**What information can you get from the *get* command?**  
*A basic view of specific resources, such as: "pods", "deployments", "services", or "namespaces"*  

### Procedure
1. Run setup for this lab

    `student@bchd:~$` `setup listing-resources-with-kubectl-get`

0. Start by ensuring you're using the `kubernetes-the-alta3-way` context.

    `student@bchd:~$` `kubectl config use-context kubernetes-the-alta3-way`  

    ```    
    Switched to context "kubernetes-the-alta3-way".
    ```

0. If you didn't switch to the `kubernetes-the-alta3-way`, it is likely that the next step will not reveal any services. The following command will list all services in the current namespace.

    `student@bchd:~$` `kubectl get services`

    ```
    NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
    Kubernetes   ClusterIP   172.16.3.1   <none>        443/TCP   4d
    ```

0. List all pods in all namespaces.

    `student@bchd:~$` `kubectl get pods --all-namespaces`

    ```
    NAMESPACE       NAME                                       READY   STATUS    RESTARTS   AGE
    kube-system     calico-kube-controllers-69cc5d4c8f-vdj77   1/1     Running   0          170m
    kube-system     calico-node-ffsvv                          1/1     Running   0          170m
    kube-system     calico-node-q5lvr                          1/1     Running   0          170m
    kube-system     kube-dns-5bc8974d44-4hjk8                  3/3     Running   0          169m
    ```

0. Curious what all those things are? A short description of each:

    - `calico-kube-controllers` - This service can be regarded as a helper container that bundles together the various components required for networking containers with Calico. The key components are: Felix, BIRD, and confd.
    - `calico-node` - This is the service on one of the 3 nodes doing networking for the pods. There are 3 nodes, so this appears 3 times.
    - `kubedns` - This is the DNS service for the cluster.

0. List all pods in the current namespace, grabbing extra details from them.     

    `student@bchd:~$` `kubectl get pods --all-namespaces -o wide`
    
    ```
    NAMESPACE       NAME                                       READY   STATUS    RESTARTS   AGE   IP               NODE              NOMINATED NODE   READINESS GATES
    kube-system     calico-kube-controllers-69cc5d4c8f-vdj77   1/1     Running   0          171m   10.14.49.232     node-1   <none>           <none>
    kube-system     calico-node-ffsvv                          1/1     Running   0          171m   10.14.49.232     node-1   <none>           <none>
    kube-system     calico-node-q5lvr                          1/1     Running   0          171m   10.7.78.19       node-2   <none>           <none>
    kube-system     kube-dns-5bc8974d44-4hjk8                  3/3     Running   0          171m   192.168.247.1    node-2   <none>           <none>
    ```
    
    > You may see different IPs listed above, than what you see. The network setup may be different for this particular class today. Either way, no worries.  

0. The `kubectl` offers lots of ways to format output:

    |     flag                                        |                                      description                          |
    |---------------------------------------------|----------------------------------------------------------------|
    | `-o=custom-columns=<spec>` | Print a table using a comma separated list of custom columns |
    | `-o=custom-columns-file=<filename>` | Print a table using the custom columns template in the `<filename>` file |
    | `-o=json` | Output a JSON formatted API object |
    | `-o=jsonpath=<template>` | Print the fields defined in a jsonpath expression |
    | `-o=jsonpath-file=<filename>` | Print the fields defined by the jsonpath expression in the `<filename>` file |
    | `-o=name` | Print only the resource name and nothing else |
    | `-o=wide` | Output in the plain-text format with any additional info and for pods, the node name is included |
    | `-o=yaml` | Output a YAML formatted API object |

0. List deployments. Currently there may not be any deployments. Shown below are examples of both.

    `student@bchd:~$` `kubectl get deployments`

    ```
    No resources found in default namespace.
    ```

    > The above is the result of no deployments being found. Or if a deployment is already running, then it will look something like this:

    ```
    NAME      DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
    nginx     1         1         1            1           4h
    ```   

0. The `--all-namespaces` flag can be used to issue across all namespaces. List all the services in the cluster, across all namespaces.

    `student@bchd:~$` `kubectl get services --all-namespaces`

    ```
    NAMESPACE     NAME         TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)                  AGE
    default       Kubernetes   ClusterIP   172.16.3.1    <none>        443/TCP                  26h
    kube-system   kube-dns     ClusterIP   172.16.3.10   <none>        53/UDP,53/TCP,9153/TCP   25h
    ```

    > `kubectl get services -A` would produce the same result -- listing all of the services across all namespaces

0. Next, run a command that shows everything except secrets. Be careful with this one in a busy cloud!

    `student@bchd:~$` `kubectl get all -A`  

    ```
    NAMESPACE     NAME                                           READY   STATUS    RESTARTS   AGE
    kube-system   pod/calico-kube-controllers-69cc5d4c8f-vdj77   1/1     Running   0          3h20m
    kube-system   pod/calico-node-ffsvv                          1/1     Running   0          3h20m
    kube-system   pod/calico-node-q5lvr                          1/1     Running   0          3h20m
    kube-system   pod/kube-dns-5bc8974d44-4hjk8                  3/3     Running   0          3h20m

    NAMESPACE     NAME                 TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)         AGE
    default       service/kubernetes   ClusterIP   172.16.3.1    <none>        443/TCP         3h22m
    kube-system   service/kube-dns     ClusterIP   172.16.3.10   <none>        53/UDP,53/TCP   3h20m

    NAMESPACE     NAME                         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR            AGE
    kube-system   daemonset.apps/calico-node   2         2         2       2            2           kubernetes.io/os=linux   3h20m

    NAMESPACE     NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
    kube-system   deployment.apps/calico-kube-controllers   1/1     1            1           3h20m
    kube-system   deployment.apps/kube-dns                  1/1     1            1           3h20m

    NAMESPACE     NAME                                                 DESIRED   CURRENT   READY   AGE
    kube-system   replicaset.apps/calico-kube-controllers-69cc5d4c8f   1         1         1       3h20m
    kube-system   replicaset.apps/kube-dns-5bc8974d44                  1         1         1       3h20m
    ```

0. Let's take a look at all of the resources for **kubeDNS**. We can do this using a **get** command with the **-f** flag to obtain all of the resources created by a manifest.

    `student@bchd:~$` `kubectl get -f k8s-config/kube-dns.yaml`

    ```
    NAME               TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)         AGE
    service/kube-dns   ClusterIP   172.16.3.10   <none>        53/UDP,53/TCP   3h59m

    NAME                      SECRETS   AGE
    serviceaccount/kube-dns   0         3h59m

    NAME                 DATA   AGE
    configmap/kube-dns   0      3h59m

    NAME                       READY   UP-TO-DATE   AVAILABLE   AGE
    deployment.apps/kube-dns   1/1     1            1           3h59m
    ```

0. Awesome work!

**CHALLENGE:**

Write `kubectl get` commands that would output the following info:

- All Pods in the `kube-system` namespace.
- All of the resources in the entire cluster that are associated with **kubedns**.  
- All Pods on `node-2`, but output also includes the Pods' IP addresses.  

<div align="center">
</div>
