# Generated by Django 3.1.4 on 2021-11-16 17:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0009_result_test'),
    ]

    operations = [
        migrations.AddField(
            model_name='test',
            name='date',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
