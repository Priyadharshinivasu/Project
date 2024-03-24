import { Component, OnInit } from '@angular/core';
import { User } from './user.model';
import { UserService } from './user.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-user',
  standalone: true,
  imports: [CommonModule,FormsModule],
  templateUrl: './user.component.html',
  styleUrl: './user.component.css'
})
export class UserComponent implements OnInit {

  constructor(private userService:UserService){}
  ngOnInit(): void {
    
    this.getUser();
  }

  newUser:User = {username:" ",email:" ",password:" "};
  userList:User[] = [];
  editingUser:User | null=null;
  updateUser:User={username:" ",email:" ",password:" "};

  createUser()
  {
    this.userService.createUser(this.newUser).subscribe(result=>{
      this.userList.push(result);
    });

    this.newUser={username:" ",email:" ",password:" "};
    
  }

  getUser()
  {
    this.userService.getUser().subscribe(result=>{
      this.userList=result;

    });
  }

  editUser(editUser:User)
  {
      this.editingUser=editUser;
      this.updateUser={...editUser};
  }

  updateUserById()
  {
    this.userService.updateUser(this.editingUser!.id!,this.updateUser).subscribe(result=>{
      const index = this.userList.findIndex((user)=>user.id == this.editingUser!.id);

      if(index!==-1)
      {
        this.userList[index]=result;
      }
    });

    this.updateUser={username:" ",email:" ",password:" "};
    
  }

  deleteUserById()
  {
    this.userService.deleteUser(this.editingUser!.id!).subscribe(()=>
    {
      this.getUser();
    }
    );
  }

}
