clear;
close all;
clc;

%% CARGAR LOS DATOS DEL VIDEO
video = VideoReader('Cilindro.mp4');
all_frames = video.read();
% MATRIZ ESTRUCTURA DE PELICULA
video_frame = struct('cdata',zeros(video.Height,video.Width,3,'uint8'));
% ANEXAMOS CADA FOTOGRAMA A LA MATRIZ
for k=1:video.NumFrames
    video_frame(k).cdata = squeeze(all_frames(:,:,:,k));
end
% SE CONVIERTE A NIVELES DE GRIS
video_frame_gray=zeros(video.Height,video.Width,video.NumFrames);
for i=1:video.NumFrames
    video_frame_gray(:,:,i) = rgb2gray(video_frame(i).cdata);
end

%% INICIALIZACION DE PARAMETROS Y COMPONENTES FILTRO DE KALMAN Y DE LA ESTRUCTURA OBJETO

n_states = 6;
n_outs = 4;
% MATRIZ A 
A  =  [1  0  1  0  0  0;
       0  1  0  1  0  0;
       0  0  1  0  0  0;
       0  0  0  1  0  0;
       0  0  0  0  1  0;
       0  0  0  0  0  1];
% MATRIZ H 
H  =  [1  0  0  0  0  0;
       0  1  0  0  0  0;
       0  0  0  0  1  0;
       0  0  0  0  0  1];
% MATRICES DE RUIDO
V = 1;
W = 1e-5;
R = 0.1*diag([0.5*V 1*V 0.1*V 0.1*V]);
Q = 10*diag([0.3*W 0.01*W 0.1*W W 10*W 10*W]);

max_obj=5; % MAXIMO NUMERO DE OBJETOS EN EL VIDEO
num_obj_presents=0; % NUMERO DE OBJETOS PRESENTES
frame_start=2;
for i=1:max_obj
    objet(i).pres=false;
    objet(i).obs=false;
    objet(i).f_no_obs=0;
    objet(i).pos=[0 0]';
    objet(i).x_hat=zeros(n_states,frame_start+video.NumFrames);
    objet(i).x_tilde=objet(i).x_hat(:,1);
    objet(i).Zk=zeros(n_outs,1);
    objet(i).P_tilde=Q; 
end

%% INICIALIZACION DE PARAMETROS CONFIGURABLES

threshold=75; % DIFERENCIA ENTRE FRAMES PARA CONSIDERAR MOVIMIENTO
minp_mov=100; % NUMERO MINIMO DE PIXELES EN MOV PARA DESCARTAR RUIDO 
maxf_occlusion=20; % NUMERO MAXIMO FRAMES PARA SUPONER OCLUSION
distmin=80; % DIST MINIMA DE PIXELES PARA SUPONER QUE DOS CLUSTERS SON EL MISMO OBJETO
distmin_est_obs=80; % DIST MINIMA DE PIXELES PARA SUPONER QUE OBJETO ESTIMADO Y OBJETO OBSERVADO SON EL MISMO OBJETO

%% BOUCLE
filter = fspecial('gaussian',20); % TIPO DE FILTRO
frame(frame_start-1).num_obj_presents = 0;

for i=frame_start:(frame_start+video.NumFrames)
    
    % DIFERENCIA DE IMAGENES
    diff_frames = video_frame_gray(:,:,i)-video_frame_gray(:,:,i-1);
    % FILTRADO
    diff_filter=filter2(filter,diff_frames,'same');
    % THRESHOLD
    frame(i).moving=(abs(diff_filter)>threshold);   
    
    % ALGORITMO KMEANS
    [row,col] = find(frame(i).moving == 1);
    xbins1 = 0:size(diff_filter,2);
    [counts,centers]=hist(col,xbins1);
    frame(i).objets_detec=0;
    frame(i).objets=0;    
    if ((~isempty(col))&&(size(col,1)>minp_mov))
        
        % CLUSTERIZADO PARA EL MAX NUM DE OBJETOS
        [agrup.idx,agrup.C,sum_clust]=kmeans(col,max_obj,'EmptyAction','singleton');
        agrup_old.idx=agrup.idx;
        
        % FUSION DE OBJETOS CERCANOS
        % MATRIZ DE DISTANCIAS ENTRE CLUSTERS
        distagrup=zeros(max_obj);
        for num=1:max_obj
            for j=num:max_obj
                distagrup(num,j)=abs(agrup.C(num)-agrup.C(j)); % DISTANCIAS ENTRE CLUSTERS
            end
        end
        distagrup=distagrup';
        distagrup_1= (distagrup<distmin).*(distagrup>0); % SE COMPARAN LAS DISTANCIAS CON LIMITES FIJADOS 
        distagrup_2=tril(distagrup_1); % SE GUARDA LA MITAD INFERIOR PUESTO QUE ES SIMETRICA
        
        obj_agrup=zeros(max_obj,max_obj);
        cluster=zeros(1,max_obj);
        num_obj=0;
        for num=1:max_obj
            if (cluster(num)==0)
                cluster(num)=1;                
                num_obj=num_obj+1; 
                obj_agrup(num_obj,num)=1; 
                % SE RECORRE LA MATRIZ EN BUSCA DE CLUSTERS CERCANOS
                for j=num+1:max_obj
                    if (distagrup_2(j,num)==1)
                        % SE INDICA COMO AGRUPADO, SE LE ASIGNA UN GRUPO Y SE BUSCAN GRUPOS CERCANOS
                        cluster(j)=1;
                        obj_agrup(num_obj,j)=1;
                        for jj=1:num
                            if (distagrup(j,jj)==1)
                                cluster(jj)=1;
                                obj_agrup(num_obj,jj)=1;
                            end
                        end
                    end
                end
            end
        end
        
        % ACTUALIZACION DE INDICES Y CENTROS
        idx_new=zeros(size(agrup.idx));
        for num=1:num_obj
            for j=1:max_obj
                if (obj_agrup(num,j)==1)
                    [row_n,col_n]=find(agrup.idx==j);
                    idx_new(row_n,col_n)=num;
                end
            end
            % CALCULO DE LOS CENTROS
            agrup.C(num)=sum(col.*(idx_new==num))/sum(idx_new==num);
        end
        agrup.idx=idx_new;
        
        % ACTUALIZACION DE NUM DE OBJETOS
        frame(i).objets=num_obj;
        
        % CALCULO DEL TAMAÑO DE LOS OBJETOS
        up   = zeros(1, frame(i).objets);
        down = zeros(1, frame(i).objets);
        right= zeros(1, frame(i).objets);
        left = zeros(1, frame(i).objets);
        for num=1:frame(i).objets
            temp=row.*(agrup.idx==num);
            temp=temp(find(temp~=0));
            down(num) = max(temp);
            up(num) = min(temp);
            temp=col.*(agrup.idx==num);
            temp=temp(find(temp~=0));
            right(num) = max(temp);
            left(num) = min(temp);
            % GUARDAR CENTROS DE OBJETOS
            frame(i).C(num,1)=(right(num)+left(num))/2;
            frame(i).C(num,2)=(down(num)+up(num))/2;
        end
    end

    % RELACION ENTRE OBJETOS ESTIMADOS Y OBSERVADOS
    frame(i).num_obj_presents=0;
    % VERIFICAR OBSERVABILIDAD DE LOS OBJETOS
    if (frame(i).objets==0)
        for j=1:max_obj
            objet(j).obs=false;
            % RESETEAR OBJETO
            if (objet(j).f_no_obs>=maxf_occlusion)
                objet(j).pres=false;
                objet(j).obs=false;
                objet(j).f_no_obs=0;
                objet(j).pos=[0 0]';
                objet(j).x_hat=zeros(n_states,frame_start+video.NumFrames);
                objet(j).x_tilde=objet(j).x_hat(:,1);
                objet(j).Zk=zeros(n_outs,1);
                objet(j).P_tilde=Q;
            end
        end 
    end
    
    for num=1:frame(i).objets
        for j=1:max_obj
            % VERIFICAR TIEMPO LIMITE DE OCLUSION
            if (objet(j).f_no_obs>=maxf_occlusion)
                % RESETEAR OBJETO
                objet(j).pres=false;
                objet(j).obs =false;
                objet(j).f_no_obs=0;
                objet(j).pos=[0 0]';
                objet(j).x_hat=zeros(n_states,frame_start+video.NumFrames);
                objet(j).x_tilde=objet(j).x_hat(:,1);
                objet(j).Zk=zeros(n_outs,1);
                objet(j).P_tilde=Q;
            end
            % MATRIZ DE DISTANCIAS ENTRE OBJETOS OBSERVADOS Y PRESENTES 
            if (objet(j).pres==true)
                dist_pres_obs(num,j)=abs(frame(i).C(num,1)-objet(j).pos(1));
            else
                dist_pres_obs(num,j)=inf;
            end
        end
        % SE COMPARAN LAS DISTANCIAS CON LIMITES FIJADOS Y SE RELACIONAN LOS OBJETOS ESTIMADOS Y OBSERVADOS
        if isempty(dist_pres_obs)
            % OBJETO ANTERIORMENTE DETECTADO PERO NO PRESENTE
            cluster_np=inf;
        else
            cluster_np=min(dist_pres_obs(num,:));
        end
         % SE COMPARAN LAS DISTANCIAS CON LIMITES FIJADOS 
         if (cluster_np<distmin_est_obs)
            % SE HACEN CORRESPONDER OBJETO PRESENTE CON OBSERVADO
            corr_obs_pres(num)=find(dist_pres_obs(num,:)==cluster_np);
            objet(corr_obs_pres(num)).obs=true;
            % ACTUALIZACION DEL OBJETO
            % SALIDA
            objet(corr_obs_pres(num)).Zk = [frame(i).C(num,1)...
                                            frame(i).C(num,2)...
                                            abs(right(num)-left(num))...
                                            abs(down(num)-up(num))]';
            % POSICION
            objet(corr_obs_pres(num)).pos= [frame(i).C(num,1)...
                                            frame(i).C(num,2)]';
            % OBSERVABILIDAD
            objet(corr_obs_pres(num)).f_no_obs=0;
        else
            % NO HAY CORRESPONDENCIA OBJETO PRESENTE CON OBSERVADO
            %non_corr_obs_pres=false;
            %cont=0;
            %while (~non_corr_obs_pres)
                %cont=cont+1;
                cont=1;
                if (objet(cont).pres==false)
                    non_corr_obs_pres=true;
                    objet(cont).pres =true;
                    objet(cont).obs  =true;
                    objet(cont).f_no_obs=0;
                    objet(cont).x_hat(:,i)=[frame(i).C(num,1)...
                                            frame(i).C(num,2)...
                                            0 0 abs(right(num)-left(num))...
                                            abs(down(num)-up(num))]';
                    objet(cont).pos=objet(cont).x_hat(1:2,i);
                    objet(cont).x_tilde=objet(cont).x_hat(:,i);
                    objet(cont).Zk=[frame(i).C(num,1)...
                                    frame(i).C(num,2)...
                                    abs(right(num)-left(num))...
                                    abs(down(num)-up(num))]';
                    objet(cont).P_tilde =Q;
                end
            %end
         end
    end 
   
    % FILTRO DE KALMAN
    for num=1:max_obj
        if (objet(num).pres && objet(num).obs)
            % GANANCIA KALMAN
            objet(num).Kk = objet(num).P_tilde*(H')/(H*objet(num).P_tilde*(H')+R);
            % ACTUALIZACION DEL VECTOR ESTADO
            objet(num).x_hat(:,i) = objet(num).x_tilde+objet(num).Kk*(objet(num).Zk-H*objet(num).x_tilde);
            % ACTUALIZACION MATRIZ COVARIANZA
            objet(num).P_hat = objet(num).P_tilde*(eye(n_states)-(H')*(objet(num).Kk'));
            % VECTOR ESTADO ESTIMADO
            objet(num).x_tilde = A*objet(num).x_hat(:,i);
            % MATRIZ COVARIANZA ESTIMADA
            objet(num).P_tilde = A*objet(num).P_hat*(A')+Q;
            objet(num).pos = objet(num).x_hat(1:2,i);
        elseif objet(num).pres
            % SE AUMENTA NUM DE FRAMES OBJETO NO OBSERVADO
            objet(num).f_no_obs = objet(num).f_no_obs+1;
            % ACTUALIZAR POSICION CON ESTIMACION
            objet(num).x_hat(:,i) = objet(num).x_tilde;
            objet(num).x_tilde = A*objet(num).x_hat(:,i);
            objet(num).pos = objet(num).x_hat(1:2,i);
        end
    end
   
    % MOSTRAR PUNTOS EN MOVIMIENTO
    figure(1);
    pause(0.01);
    imshow(uint8(video_frame_gray(:,:,i))); 
    hold on;
    color=['r','b','y','g','k'];
    for j=1:max_obj
        if objet(j).pres
            if ((objet(j).x_hat(5,i)~=0)&&(objet(j).x_hat(6,i)~=0))
                rectangle('pos',[(objet(j).x_hat(1,i)-objet(j).x_hat(5,i)/2) (objet(j).x_hat(2,i)-objet(j).x_hat(6,i)/2) objet(j).x_hat(5,i) objet(j).x_hat(6,i)],'EdgeColor',color(j),'Linewidth',3);
                strmax = ['Objeto ',num2str(j)];
                text((objet(j).x_hat(1,i)-objet(j).x_hat(5,i)/2),(objet(j).x_hat(2,i)-objet(j).x_hat(6,i)/2-20),strmax,'HorizontalAlignment','left', 'Color',color(j));
            end
        end
    end
    objets=sum([objet(1:end).pres]);
    title(sprintf('Objects in the video => %i',objets));
end