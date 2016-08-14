merge = """
drop table if exists {table_name}_tmp;
create table {table_name}_tmp as 
select c.artist_id,c.ds,c.artist_target,
artist_song_listened_num_one_month_train.level_1_artist_song_listened_num ,
artist_song_listened_num_one_month_train.level_2_artist_song_listened_num ,
artist_song_listened_num_one_month_train.level_3_artist_song_listened_num ,
artist_song_listened_num_one_month_train.level_4_artist_song_listened_num ,
artist_song_listened_num_one_month_train.level_5_artist_song_listened_num
,
artist_song_num_one_month_train.artist_song_num
,
artist_duplicated_song_one_month_train.artist_duplicated_song_num ,
artist_duplicated_song_one_month_train.artist_duplicated_song_rate
,
artist_gender_one_month_train.artist_gender ,
artist_language_num_one_month_train.artist_language
,
artist_main_language_one_month_train.artist_main_language ,
artist_init_plays_one_month_train.max_artist_init_plays ,
artist_init_plays_one_month_train.min_artist_init_plays ,
artist_init_plays_one_month_train.harmonic_avg_aritst_init_plays ,
artist_init_plays_one_month_train.var_artist_init_plays ,
artist_init_plays_one_month_train.ten_per_artist_init_plays ,
artist_init_plays_one_month_train.one_quartile_artist_init_plays ,
artist_init_plays_one_month_train.median_artist_init_plays ,
artist_init_plays_one_month_train.three_quartile_artist_init_plays ,
artist_init_plays_one_month_train.ninty_per_artist_init_plays ,
artist_init_plays_one_month_train.avg_artist_init_plays
,
artist_song_play_one_month_train.avg_play_num ,
artist_song_play_one_month_train.min_play_num ,
artist_song_play_one_month_train.max_play_num ,
artist_song_play_one_month_train.harmonic_avg_play_num ,
artist_song_play_one_month_train.var_play_num ,
artist_song_play_one_month_train.ten_per_play_num ,
artist_song_play_one_month_train.one_quartile_play_num ,
artist_song_play_one_month_train.median_quartile_play_num ,
artist_song_play_one_month_train.three_quartile_play_num ,
artist_song_play_one_month_train.ninty_per_play_num
,
artist_song_download_one_month_train.avg_down_num ,
artist_song_download_one_month_train.min_down_num ,
artist_song_download_one_month_train.max_down_num ,
artist_song_download_one_month_train.harmonic_avg_down_num ,
artist_song_download_one_month_train.var_down_num ,
artist_song_download_one_month_train.ten_per_down_num ,
artist_song_download_one_month_train.one_quartile_down_num ,
artist_song_download_one_month_train.median_down_num ,
artist_song_download_one_month_train.three_quartile_down_num ,
artist_song_download_one_month_train.ninty_per_down_num
,
artist_song_save_one_month_train.avg_save_num ,
artist_song_save_one_month_train.min_save_num ,
artist_song_save_one_month_train.max_save_num ,
artist_song_save_one_month_train.harmonic_avg_save_num ,
artist_song_save_one_month_train.var_save_num ,
artist_song_save_one_month_train.ten_per_save_num ,
artist_song_save_one_month_train.one_quartile_save_num ,
artist_song_save_one_month_train.median_quartile_save_num ,
artist_song_save_one_month_train.three_quartile_save_num ,
artist_song_save_one_month_train.ninty_per_save_num
,
artist_fans_one_month_train.fans_num ,
artist_fans_one_month_train.level_1_fans_num ,
artist_fans_one_month_train.level_2_fans_num ,
artist_fans_one_month_train.level_3_fans_num ,
artist_fans_one_month_train.level_4_fans_num ,
artist_fans_one_month_train.level_5_fans_num ,
artist_fans_one_month_train.level_1_fans_num_ratio ,
artist_fans_one_month_train.level_2_fans_num_ratio ,
artist_fans_one_month_train.level_3_fans_num_ratio ,
artist_fans_one_month_train.level_4_fans_num_ratio ,
artist_fans_one_month_train.level_5_fans_num_ratio
,
artist_song_fans_one_month_train.level_1_fans_num_song ,
artist_song_fans_one_month_train.level_2_fans_num_song ,
artist_song_fans_one_month_train.level_3_fans_num_song ,
artist_song_fans_one_month_train.level_4_fans_num_song ,
artist_song_fans_one_month_train.level_5_fans_num_song ,
artist_song_fans_one_month_train.level_1_fans_num_song_ratio ,
artist_song_fans_one_month_train.level_2_fans_num_song_ratio ,
artist_song_fans_one_month_train.level_3_fans_num_song_ratio ,
artist_song_fans_one_month_train.level_4_fans_num_song_ratio ,
artist_song_fans_one_month_train.level_5_fans_num_song_ratio
,
artist_album_one_month_train.artist_album
,
artist_album_frequency_one_month_train.artist_album_frequency
,
artist_song_frequency_one_month_train.artist_song_frequency
from artist_target_one_month_train c
left outer join artist_song_listened_num_one_month_train on(c.artist_id = artist_song_listened_num_one_month_train.artist_id )
left outer join artist_song_num_one_month_train on(c.artist_id = artist_song_num_one_month_train.artist_id )
left outer join artist_duplicated_song_one_month_train on(c.artist_id = artist_duplicated_song_one_month_train.artist_id )
left outer join artist_gender_one_month_train on(c.artist_id = artist_gender_one_month_train.artist_id )
left outer join artist_language_num_one_month_train on(c.artist_id = artist_language_num_one_month_train.artist_id )
left outer join artist_main_language_one_month_train on(c.artist_id = artist_main_language_one_month_train.artist_id )
left outer join artist_init_plays_one_month_train on(c.artist_id = artist_init_plays_one_month_train.artist_id )
left outer join artist_song_play_one_month_train on(c.artist_id = artist_song_play_one_month_train.artist_id )
left outer join artist_song_download_one_month_train on(c.artist_id = artist_song_download_one_month_train.artist_id )
left outer join artist_song_save_one_month_train on(c.artist_id = artist_song_save_one_month_train.artist_id )
left outer join artist_fans_one_month_train on(c.artist_id = artist_fans_one_month_train.artist_id )
left outer join artist_song_fans_one_month_train on(c.artist_id = artist_song_fans_one_month_train.artist_id )
left outer join artist_album_one_month_train on(c.artist_id = artist_album_one_month_train.artist_id )
left outer join artist_album_frequency_one_month_train on(c.artist_id = artist_album_frequency_one_month_train.artist_id )
left outer join artist_song_frequency_one_month_train on(c.artist_id = artist_song_frequency_one_month_train.artist_id )
;




drop table if exists {table_name};
create table {table_name} as 
select c.*,
artist_play_static_one_month_train.three_day_dis_song_play ,
artist_play_static_one_month_train.three_day_song_play ,
artist_play_static_one_month_train.fifteen_day_dis_song_play ,
artist_play_static_one_month_train.fifteen_day_song_play ,
artist_play_static_one_month_train.thirty_day_dis_song_play ,
artist_play_static_one_month_train.thirty_day_song_play ,
artist_play_static_one_month_train.three_day_dis_user_play ,
artist_play_static_one_month_train.three_day_user_play ,
artist_play_static_one_month_train.fifteen_day_dis_user_play ,
artist_play_static_one_month_train.fifteen_day_user_play ,
artist_play_static_one_month_train.thirty_day_user_play ,
artist_play_static_one_month_train.thirty_day_dis_user_play
,
artist_down_static_one_month_train.three_day_dis_song_down ,
artist_down_static_one_month_train.three_day_song_down ,
artist_down_static_one_month_train.fifteen_day_dis_song_down ,
artist_down_static_one_month_train.fifteen_day_song_down ,
artist_down_static_one_month_train.thirty_day_dis_song_down ,
artist_down_static_one_month_train.thirty_day_song_down ,
artist_down_static_one_month_train.three_day_dis_user_down ,
artist_down_static_one_month_train.three_day_user_down ,
artist_down_static_one_month_train.fifteen_day_dis_user_down ,
artist_down_static_one_month_train.fifteen_day_user_down ,
artist_down_static_one_month_train.thirty_day_user_down ,
artist_down_static_one_month_train.thirty_day_dis_user_down
,
artist_save_static_one_month_train.three_day_dis_song_save ,
artist_save_static_one_month_train.three_day_song_save ,
artist_save_static_one_month_train.fifteen_day_dis_song_save ,
artist_save_static_one_month_train.fifteen_day_song_save ,
artist_save_static_one_month_train.thirty_day_dis_song_save ,
artist_save_static_one_month_train.thirty_day_song_save ,
artist_save_static_one_month_train.three_day_dis_user_save ,
artist_save_static_one_month_train.three_day_user_save ,
artist_save_static_one_month_train.fifteen_day_dis_user_save ,
artist_save_static_one_month_train.fifteen_day_user_save ,
artist_save_static_one_month_train.thirty_day_user_save ,
artist_save_static_one_month_train.thirty_day_dis_user_save
,
artist_sequence_ratio_one_month_train.leve1_play_ratio ,
artist_sequence_ratio_one_month_train.leve2_play_ratio ,
artist_sequence_ratio_one_month_train.leve1_play_song_ratio ,
artist_sequence_ratio_one_month_train.leve2_play_song_ratio ,
artist_sequence_ratio_one_month_train.leve1_play_user_ratio ,
artist_sequence_ratio_one_month_train.leve2_play_user_ratio ,
artist_sequence_ratio_one_month_train.leve1_save_ratio ,
artist_sequence_ratio_one_month_train.leve2_save_ratio ,
artist_sequence_ratio_one_month_train.leve1_save_song_ratio ,
artist_sequence_ratio_one_month_train.leve2_save_song_ratio ,
artist_sequence_ratio_one_month_train.leve1_save_user_ratio ,
artist_sequence_ratio_one_month_train.leve2_save_user_ratio ,
artist_sequence_ratio_one_month_train.leve1_down_ratio ,
artist_sequence_ratio_one_month_train.leve2_down_ratio ,
artist_sequence_ratio_one_month_train.leve1_down_song_ratio ,
artist_sequence_ratio_one_month_train.leve2_down_song_ratio ,
artist_sequence_ratio_one_month_train.leve1_down_user_ratio ,
artist_sequence_ratio_one_month_train.leve2_down_user_ratio
,
artist_song_play_static_one_month_train.avg_artist_song_play ,
artist_song_play_static_one_month_train.max_artist_song_play ,
artist_song_play_static_one_month_train.min_artist_song_play ,
artist_song_play_static_one_month_train.var_artist_song_plays ,
artist_song_play_static_one_month_train.harmonic_avg_song_play ,
artist_song_play_static_one_month_train.one_quartile_artist_song_play ,
artist_song_play_static_one_month_train.median_artist_song_play ,
artist_song_play_static_one_month_train.three_quartile_artist_song_play ,
artist_song_play_static_one_month_train.ninty_per_artist_song_play
,
artist_popular_song_one_month_train.song_num ,
artist_popular_song_one_month_train.level_1_song_num ,
artist_popular_song_one_month_train.level_2_song_num ,
artist_popular_song_one_month_train.level_3_song_num ,
artist_popular_song_one_month_train.level_4_song_num ,
artist_popular_song_one_month_train.level_5_song_num ,
artist_popular_song_one_month_train.level_6_song_num
,
artist_popular_song_user_one_month_train.level_1_song_user ,
artist_popular_song_user_one_month_train.level_2_song_user ,
artist_popular_song_user_one_month_train.level_3_song_user ,
artist_popular_song_user_one_month_train.level_4_song_user ,
artist_popular_song_user_one_month_train.level_5_song_user ,
artist_popular_song_user_one_month_train.level_6_song_user
,
artist_listen_days_one_month_train.artist_listen_days
,
artist_play_one_month_train.artist_play_1 ,
artist_play_one_month_train.artist_play_2 ,
artist_play_one_month_train.artist_play_3 ,
artist_play_one_month_train.artist_play_4 ,
artist_play_one_month_train.artist_play_5 ,
artist_play_one_month_train.artist_play_6 ,
artist_play_one_month_train.artist_play_7 ,
artist_play_one_month_train.artist_play_8 ,
artist_play_one_month_train.artist_play_9 ,
artist_play_one_month_train.artist_play_10 ,
artist_play_one_month_train.artist_play_11 ,
artist_play_one_month_train.artist_play_12 ,
artist_play_one_month_train.artist_play_13 ,
artist_play_one_month_train.artist_play_14 ,
artist_play_one_month_train.artist_play_15 ,
artist_play_one_month_train.artist_play_16 ,
artist_play_one_month_train.artist_play_17 ,
artist_play_one_month_train.artist_play_18 ,
artist_play_one_month_train.artist_play_19 ,
artist_play_one_month_train.artist_play_20 ,
artist_play_one_month_train.artist_play_21 ,
artist_play_one_month_train.artist_play_22 ,
artist_play_one_month_train.artist_play_23 ,
artist_play_one_month_train.artist_play_24 ,
artist_play_one_month_train.artist_play_25 ,
artist_play_one_month_train.artist_play_26 ,
artist_play_one_month_train.artist_play_27 ,
artist_play_one_month_train.artist_play_28 ,
artist_play_one_month_train.artist_play_29 ,
artist_play_one_month_train.artist_play_30
,
artist_save_one_month_train.artist_save_1 ,
artist_save_one_month_train.artist_save_2 ,
artist_save_one_month_train.artist_save_3 ,
artist_save_one_month_train.artist_save_4 ,
artist_save_one_month_train.artist_save_5 ,
artist_save_one_month_train.artist_save_6 ,
artist_save_one_month_train.artist_save_7 ,
artist_save_one_month_train.artist_save_8 ,
artist_save_one_month_train.artist_save_9 ,
artist_save_one_month_train.artist_save_10 ,
artist_save_one_month_train.artist_save_11 ,
artist_save_one_month_train.artist_save_12 ,
artist_save_one_month_train.artist_save_13 ,
artist_save_one_month_train.artist_save_14 ,
artist_save_one_month_train.artist_save_15 ,
artist_save_one_month_train.artist_save_16 ,
artist_save_one_month_train.artist_save_17 ,
artist_save_one_month_train.artist_save_18 ,
artist_save_one_month_train.artist_save_19 ,
artist_save_one_month_train.artist_save_20 ,
artist_save_one_month_train.artist_save_21 ,
artist_save_one_month_train.artist_save_22 ,
artist_save_one_month_train.artist_save_23 ,
artist_save_one_month_train.artist_save_24 ,
artist_save_one_month_train.artist_save_25 ,
artist_save_one_month_train.artist_save_26 ,
artist_save_one_month_train.artist_save_27 ,
artist_save_one_month_train.artist_save_28 ,
artist_save_one_month_train.artist_save_29 ,
artist_save_one_month_train.artist_save_30
,
artist_down_one_month_train.artist_down_1 ,
artist_down_one_month_train.artist_down_2 ,
artist_down_one_month_train.artist_down_3 ,
artist_down_one_month_train.artist_down_4 ,
artist_down_one_month_train.artist_down_5 ,
artist_down_one_month_train.artist_down_6 ,
artist_down_one_month_train.artist_down_7 ,
artist_down_one_month_train.artist_down_8 ,
artist_down_one_month_train.artist_down_9 ,
artist_down_one_month_train.artist_down_10 ,
artist_down_one_month_train.artist_down_11 ,
artist_down_one_month_train.artist_down_12 ,
artist_down_one_month_train.artist_down_13 ,
artist_down_one_month_train.artist_down_14 ,
artist_down_one_month_train.artist_down_15 ,
artist_down_one_month_train.artist_down_16 ,
artist_down_one_month_train.artist_down_17 ,
artist_down_one_month_train.artist_down_18 ,
artist_down_one_month_train.artist_down_19 ,
artist_down_one_month_train.artist_down_20 ,
artist_down_one_month_train.artist_down_21 ,
artist_down_one_month_train.artist_down_22 ,
artist_down_one_month_train.artist_down_23 ,
artist_down_one_month_train.artist_down_24 ,
artist_down_one_month_train.artist_down_25 ,
artist_down_one_month_train.artist_down_26 ,
artist_down_one_month_train.artist_down_27 ,
artist_down_one_month_train.artist_down_28 ,
artist_down_one_month_train.artist_down_29 ,
artist_down_one_month_train.artist_down_30
from 
{table_name}_tmp c
left outer join artist_play_static_one_month_train on(c.artist_id = artist_play_static_one_month_train.artist_id )
left outer join artist_down_static_one_month_train on(c.artist_id = artist_down_static_one_month_train.artist_id )
left outer join artist_save_static_one_month_train on(c.artist_id = artist_save_static_one_month_train.artist_id )
left outer join artist_sequence_ratio_one_month_train on(c.artist_id = artist_sequence_ratio_one_month_train.artist_id )
left outer join artist_song_play_static_one_month_train on(c.artist_id = artist_song_play_static_one_month_train.artist_id )
left outer join artist_popular_song_one_month_train on(c.artist_id = artist_popular_song_one_month_train.artist_id )
left outer join artist_popular_song_user_one_month_train on(c.artist_id = artist_popular_song_user_one_month_train.artist_id )
left outer join artist_listen_days_one_month_train on(c.artist_id = artist_listen_days_one_month_train.artist_id )
left outer join artist_play_one_month_train on(c.artist_id = artist_play_one_month_train.artist_id )
left outer join artist_save_one_month_train on(c.artist_id = artist_save_one_month_train.artist_id )
left outer join artist_down_one_month_train on(c.artist_id = artist_down_one_month_train.artist_id )
;

"""
"""
create table one_month_all as
select *
from
(
    select *, 0 as tag from one_month_train
    union all 
    select *, 1 as tag from one_month_test
    union all 
    select *, 2 as tag from one_month_submit
)a
;
"""
import datetime
def print_to_file(predict_day, gene_file, union_select_all):
    gap_day = predict_day + datetime.timedelta(days = -1)
    split_day_begin = predict_day + datetime.timedelta(days = -31)
    print>>gene_file, sql_code.format(predict_day = predict_day.strftime("%Y%m%d"),
        gap_day = gap_day.strftime("%Y%m%d"),
        gap_day_format=gap_day.strftime("%Y-%m-%d"),
        split_day_begin=split_day_begin.strftime("%Y%m%d"),
        split_last_day_format=split_day_begin.strftime("%Y-%m-%d"))
    print>>gene_file, merge.format(table_name = "train_%s" %(predict_day.strftime("%Y%m%d"),))
    union_select_all.append("\tselect * from train_%s\n" %(predict_day.strftime("%Y%m%d"),))
def get_day_by_str(day):
    return datetime.date(int(day[:4]), int(day[4:6]), int(day[6:8]))
if __name__ == '__main__':
    sql_code = ""
    union_select_all = []
    with open("ynh_train.sql") as sql:
        for line in sql:
            sql_code += line
    predict_day_begin = "20150502"
    predict_day = get_day_by_str(predict_day_begin)
    gene_file = open("tmp", "w")
    for i in xrange(0, 30):
        if i % 2 == 0:
            gene_file.close()
            gene_file = open("gen_fea_day_%d_%d.sql" %(i+2,i+3), "w")
        print_to_file(predict_day, gene_file, union_select_all)
        predict_day = predict_day + datetime.timedelta(days = 1)
    
    gene_file.close()
    predict_day = get_day_by_str("20150702")
    gene_file = open("gen_fea_day_test.sql", "w")
    print_to_file(predict_day, gene_file, [])
    predict_day = get_day_by_str("20150901")
    gene_file = open("gen_fea_day_submit.sql", "w")
    print_to_file(predict_day, gene_file, [])
    union_file = open("union.sql", "w")
    print>>union_file, "create table train_all as\nselect * from\n(\n{union_select_all}\n)a;".format(union_select_all="\tunion all\n".join(union_select_all))
    union_file.close()