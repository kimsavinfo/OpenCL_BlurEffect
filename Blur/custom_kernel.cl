#define DIM 800 // longueur et largeur du canevas = DIM * DIM pixels
#define WORKGROUP_DIM 10 // nombre d'items dans un workgroup = NB_WORK_ITEM_LINE * NB_WORK_ITEM_LINE

__kernel
void custom_kernel(  __global unsigned char *ptr )
{
    // On initialise le chache : stockera total des ARGB
    __local int cache[4];
    cache[0] = 0;
    cache[1] = 0;
    cache[2] = 0;
    cache[3] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    int nbWorkGroup = (DIM * DIM) / (WORKGROUP_DIM * WORKGROUP_DIM);
    int nbWorkGroupParLigne = DIM / WORKGROUP_DIM;
    int nbItemsParWorkGroup = WORKGROUP_DIM * WORKGROUP_DIM;
    int iWorkGroup = get_group_id(0);
    int iLocalId = get_local_id(0);
    
    int iLigneWG = iWorkGroup / nbWorkGroupParLigne;
    int iColonneWG = iWorkGroup - iLigneWG * nbWorkGroupParLigne;
    
    int iLigneLocale = iLocalId / WORKGROUP_DIM;
    int iColonneLocale = iLocalId - iLigneLocale * WORKGROUP_DIM;
    
    int ligne = iLigneWG * (DIM * WORKGROUP_DIM) + iLigneLocale * DIM;
    int colonne = iColonneWG * WORKGROUP_DIM + iColonneLocale;
    
    int iPtr = ligne + colonne;
    int offset = iPtr * 4;
    
    
    cache[0] += ptr[offset];
    cache[1] += ptr[offset + 1];
    cache[2] += ptr[offset + 2];
    cache[3] += ptr[offset + 3];
    
    // On attend d'avoir récupérer le total des ARGB de chaque pixel
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // On attribue la moyenne à tous les work-items du work-group
    ptr[offset] = cache[0] ; // / nbItemsParWorkGroup;
    ptr[offset + 1] = cache[1] ; //  / nbItemsParWorkGroup;
    ptr[offset + 2] = cache[2] ; //  / nbItemsParWorkGroup;
    ptr[offset + 3] = cache[3] ; //  / nbItemsParWorkGroup;
    
}